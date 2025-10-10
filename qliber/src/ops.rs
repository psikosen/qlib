use std::collections::VecDeque;
use std::fmt;
use std::str::FromStr;

use polars::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("unknown identifier `{0}`")]
    UnknownIdentifier(String),
    #[error("mismatched parenthesis in expression")]
    UnbalancedParenthesis,
    #[error("unexpected end of expression")]
    UnexpectedEof,
    #[error("unexpected token `{0}`")]
    UnexpectedToken(String),
    #[error("rolling window must be positive")]
    InvalidWindow,
    #[error("lag periods must be non-negative")]
    InvalidLag,
    #[error("percentile must be between 0 and 1 inclusive")]
    InvalidPercentile,
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("series casting error: {0}")]
    Casting(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Number,
    Ident,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokenKind,
    lexeme: String,
}

fn tokenize(input: &str) -> Result<Vec<Token>, OpsError> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '0'..='9' | '.' => {
                let mut literal = String::new();
                let mut seen_dot = ch == '.';
                let mut seen_exp = false;
                while let Some(&c) = chars.peek() {
                    match c {
                        '0'..='9' => {
                            literal.push(c);
                            chars.next();
                        }
                        '.' if !seen_dot => {
                            seen_dot = true;
                            literal.push(c);
                            chars.next();
                        }
                        'e' | 'E' if !seen_exp => {
                            seen_exp = true;
                            literal.push(c);
                            chars.next();
                            if let Some(&sign) = chars.peek()
                                && (sign == '+' || sign == '-')
                            {
                                literal.push(sign);
                                chars.next();
                            }
                        }
                        _ => break,
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Number,
                    lexeme: literal,
                });
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Ident,
                    lexeme: ident,
                });
            }
            '+' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Plus,
                    lexeme: "+".into(),
                });
            }
            '-' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Minus,
                    lexeme: "-".into(),
                });
            }
            '*' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Star,
                    lexeme: "*".into(),
                });
            }
            '/' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Slash,
                    lexeme: "/".into(),
                });
            }
            '^' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Caret,
                    lexeme: "^".into(),
                });
            }
            '(' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::LParen,
                    lexeme: "(".into(),
                });
            }
            ')' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::RParen,
                    lexeme: ")".into(),
                });
            }
            ',' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Comma,
                    lexeme: ",".into(),
                });
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            other => return Err(OpsError::UnexpectedToken(other.to_string())),
        }
    }

    Ok(tokens)
}

#[derive(Debug, Clone)]
enum ExprNode {
    Column(String),
    Literal(f64),
    UnaryNeg(Box<ExprNode>),
    Binary {
        op: BinaryOp,
        left: Box<ExprNode>,
        right: Box<ExprNode>,
    },
    RollingMean {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingStd {
        expr: Box<ExprNode>,
        window: usize,
    },
    ExpandingSum(Box<ExprNode>),
    Percentile {
        expr: Box<ExprNode>,
        quantile: f64,
    },
    Lag {
        expr: Box<ExprNode>,
        periods: usize,
    },
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone)]
pub struct Expression {
    root: ExprNode,
}

impl Expression {
    pub fn evaluate(&self, frame: &DataFrame) -> Result<Series, OpsError> {
        evaluate(&self.root, frame)
    }
}

impl FromStr for Expression {
    type Err = OpsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = tokenize(s)?;
        let mut parser = Parser::new(tokens);
        let node = parser.parse_expression()?;
        if parser.position < parser.tokens.len() {
            return Err(OpsError::UnexpectedToken(
                parser.tokens[parser.position].lexeme.clone(),
            ));
        }
        Ok(Self { root: node })
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn peek_kind(&self) -> Option<TokenKind> {
        self.tokens.get(self.position).map(|token| token.kind)
    }

    fn next_token(&mut self) -> Result<Token, OpsError> {
        let token = self
            .tokens
            .get(self.position)
            .cloned()
            .ok_or(OpsError::UnexpectedEof)?;
        self.position += 1;
        Ok(token)
    }

    fn match_kind(&mut self, kind: TokenKind) -> bool {
        if self.peek_kind() == Some(kind) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn parse_expression(&mut self) -> Result<ExprNode, OpsError> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_mul_div()?;
        loop {
            match self.peek_kind() {
                Some(TokenKind::Plus) => {
                    self.position += 1;
                    let rhs = self.parse_mul_div()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::Minus) => {
                    self.position += 1;
                    let rhs = self.parse_mul_div()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Sub,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_mul_div(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_pow()?;
        loop {
            match self.peek_kind() {
                Some(TokenKind::Star) => {
                    self.position += 1;
                    let rhs = self.parse_pow()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::Slash) => {
                    self.position += 1;
                    let rhs = self.parse_pow()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Div,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_pow(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_unary()?;
        loop {
            if self.peek_kind() == Some(TokenKind::Caret) {
                self.position += 1;
                let rhs = self.parse_unary()?;
                node = ExprNode::Binary {
                    op: BinaryOp::Pow,
                    left: Box::new(node),
                    right: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_unary(&mut self) -> Result<ExprNode, OpsError> {
        if self.match_kind(TokenKind::Minus) {
            let expr = self.parse_unary()?;
            Ok(ExprNode::UnaryNeg(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<ExprNode, OpsError> {
        let token = self.next_token()?;
        match token.kind {
            TokenKind::Number => {
                let value: f64 = token
                    .lexeme
                    .parse()
                    .map_err(|_| OpsError::UnexpectedToken(token.lexeme.clone()))?;
                Ok(ExprNode::Literal(value))
            }
            TokenKind::Ident => {
                if self.peek_kind() == Some(TokenKind::LParen) {
                    self.consume_function(token)
                } else {
                    Ok(ExprNode::Column(token.lexeme.clone()))
                }
            }
            TokenKind::LParen => {
                let expr = self.parse_expression()?;
                if !self.match_kind(TokenKind::RParen) {
                    return Err(OpsError::UnbalancedParenthesis);
                }
                Ok(expr)
            }
            _ => Err(OpsError::UnexpectedToken(token.lexeme.clone())),
        }
    }

    fn consume_function(&mut self, name_token: Token) -> Result<ExprNode, OpsError> {
        self.position += 1; // consume '('
        let mut args = Vec::new();
        if self.peek_kind() != Some(TokenKind::RParen) {
            loop {
                args.push(self.parse_expression()?);
                if self.match_kind(TokenKind::Comma) {
                    continue;
                }
                break;
            }
        }
        if !self.match_kind(TokenKind::RParen) {
            return Err(OpsError::UnbalancedParenthesis);
        }

        let func = name_token.lexeme.to_lowercase();
        match func.as_str() {
            "rolling_mean" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_mean expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingMean {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            "rolling_std" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_std expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingStd {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            "expanding_sum" => {
                if args.len() != 1 {
                    return Err(OpsError::UnexpectedToken(
                        "expanding_sum expects one argument".into(),
                    ));
                }
                Ok(ExprNode::ExpandingSum(Box::new(args[0].clone())))
            }
            "percentile" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "percentile expects two arguments".into(),
                    ));
                }
                let quantile = expect_literal_f64(&args[1])?;
                if !(0.0..=1.0).contains(&quantile) {
                    return Err(OpsError::InvalidPercentile);
                }
                Ok(ExprNode::Percentile {
                    expr: Box::new(args[0].clone()),
                    quantile,
                })
            }
            "lag" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "lag expects two arguments".into(),
                    ));
                }
                let periods = expect_literal_usize(&args[1])?;
                Ok(ExprNode::Lag {
                    expr: Box::new(args[0].clone()),
                    periods,
                })
            }
            _ => Err(OpsError::UnknownIdentifier(name_token.lexeme.clone())),
        }
    }
}

fn expect_literal_usize(node: &ExprNode) -> Result<usize, OpsError> {
    match node {
        ExprNode::Literal(value) => {
            if *value < 0.0 {
                return Err(OpsError::UnexpectedToken(format!("{value}")));
            }
            Ok(*value as usize)
        }
        _ => Err(OpsError::UnexpectedToken(
            "expected literal integer argument".into(),
        )),
    }
}

fn expect_literal_f64(node: &ExprNode) -> Result<f64, OpsError> {
    match node {
        ExprNode::Literal(value) => Ok(*value),
        _ => Err(OpsError::UnexpectedToken(
            "expected literal numeric argument".into(),
        )),
    }
}

fn ensure_f64(series: Series) -> Result<Float64Chunked, OpsError> {
    if series.dtype() == &DataType::Float64 {
        Ok(series
            .f64()
            .map_err(|_| OpsError::Casting("failed to access f64 series".into()))?
            .clone())
    } else {
        let casted = series.cast(&DataType::Float64).map_err(OpsError::Polars)?;
        Ok(casted
            .f64()
            .map_err(|_| OpsError::Casting("failed to cast series to f64".into()))?
            .clone())
    }
}

fn evaluate(node: &ExprNode, frame: &DataFrame) -> Result<Series, OpsError> {
    match node {
        ExprNode::Column(name) => frame
            .column(name)
            .map_err(|_| OpsError::UnknownIdentifier(name.clone()))
            .cloned(),
        ExprNode::Literal(value) => {
            Ok(Float64Chunked::full("literal", *value, frame.height()).into_series())
        }
        ExprNode::UnaryNeg(expr) => {
            let evaluated = ensure_f64(evaluate(expr, frame)?)?;
            let negated: Float64Chunked = evaluated
                .into_iter()
                .map(|value| value.map(|v| -v))
                .collect();
            Ok(negated.into_series())
        }
        ExprNode::Binary { op, left, right } => {
            let lhs = ensure_f64(evaluate(left, frame)?)?;
            let rhs = ensure_f64(evaluate(right, frame)?)?;
            let rhs_iter = rhs.into_iter();
            let values: Float64Chunked = lhs
                .into_iter()
                .zip(rhs_iter)
                .map(|(l, r)| match (op, l, r) {
                    (_, None, _) | (_, _, None) => None,
                    (BinaryOp::Add, Some(l), Some(r)) => Some(l + r),
                    (BinaryOp::Sub, Some(l), Some(r)) => Some(l - r),
                    (BinaryOp::Mul, Some(l), Some(r)) => Some(l * r),
                    (BinaryOp::Div, Some(l), Some(r)) => Some(l / r),
                    (BinaryOp::Pow, Some(l), Some(r)) => Some(l.powf(r)),
                })
                .collect();
            Ok(values.into_series())
        }
        ExprNode::RollingMean { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_mean(&base, *window).into_series())
        }
        ExprNode::RollingStd { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_std(&base, *window).into_series())
        }
        ExprNode::ExpandingSum(expr) => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(expanding_sum(&base).into_series())
        }
        ExprNode::Percentile { expr, quantile } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(
                Float64Chunked::full("percentile", percentile(&base, *quantile), base.len())
                    .into_series(),
            )
        }
        ExprNode::Lag { expr, periods } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(lag(&base, *periods).into_series())
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expression")
    }
}

pub fn evaluate_expression(expr: &str, frame: &DataFrame) -> Result<Series, OpsError> {
    let expression = Expression::from_str(expr)?;
    expression.evaluate(frame)
}

fn rolling_mean(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    let mut sum = 0.0;
    let mut count = 0.0;
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if let Some(v) = value {
                sum += v;
                count += 1.0;
            }
            if queue.len() > window
                && let Some(old) = queue.pop_front().flatten()
            {
                sum -= old;
                count -= 1.0;
            }
            if count > 0.0 { Some(sum / count) } else { None }
        })
        .collect()
}

fn rolling_std(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0.0;
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if let Some(v) = value {
                sum += v;
                sum_sq += v * v;
                count += 1.0;
            }
            if queue.len() > window
                && let Some(old) = queue.pop_front().flatten()
            {
                sum -= old;
                sum_sq -= old * old;
                count -= 1.0;
            }
            if count > 1.0 {
                let mean = sum / count;
                let variance = (sum_sq / count) - mean * mean;
                Some(variance.max(0.0).sqrt())
            } else {
                Some(0.0)
            }
        })
        .collect()
}

fn expanding_sum(values: &Float64Chunked) -> Float64Chunked {
    let mut total = 0.0;
    values
        .into_iter()
        .map(|value| match value {
            Some(v) => {
                total += v;
                Some(total)
            }
            None => Some(total),
        })
        .collect()
}

fn percentile(values: &Float64Chunked, quantile: f64) -> f64 {
    let mut collected: Vec<f64> = values.into_iter().flatten().collect();
    if collected.is_empty() {
        return 0.0;
    }
    collected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((collected.len() - 1) as f64 * quantile).round() as usize;
    collected[idx.min(collected.len() - 1)]
}

fn lag(values: &Float64Chunked, periods: usize) -> Float64Chunked {
    let mut buffer = Vec::with_capacity(values.len());
    for _ in 0..periods {
        buffer.push(None);
    }
    let iter = values.into_iter();
    buffer.extend(iter);
    buffer.truncate(values.len());
    buffer.into_iter().collect()
}
