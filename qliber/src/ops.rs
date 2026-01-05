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
    Greater,
    Less,
    GreaterEq,
    LessEq,
    EqEq,
    NotEq,
    And,
    Or,
    Not,
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
            '>' => {
                chars.next();
                if chars.peek() == Some(&'=') {
                    chars.next();
                    tokens.push(Token {
                        kind: TokenKind::GreaterEq,
                        lexeme: ">=".into(),
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Greater,
                        lexeme: ">".into(),
                    });
                }
            }
            '<' => {
                chars.next();
                if chars.peek() == Some(&'=') {
                    chars.next();
                    tokens.push(Token {
                        kind: TokenKind::LessEq,
                        lexeme: "<=".into(),
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Less,
                        lexeme: "<".into(),
                    });
                }
            }
            '=' => {
                chars.next();
                if chars.peek() == Some(&'=') {
                    chars.next();
                    tokens.push(Token {
                        kind: TokenKind::EqEq,
                        lexeme: "==".into(),
                    });
                } else {
                    return Err(OpsError::UnexpectedToken("=".into()));
                }
            }
            '!' => {
                chars.next();
                if chars.peek() == Some(&'=') {
                    chars.next();
                    tokens.push(Token {
                        kind: TokenKind::NotEq,
                        lexeme: "!=".into(),
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Not,
                        lexeme: "!".into(),
                    });
                }
            }
            '&' => {
                chars.next();
                if chars.peek() == Some(&'&') {
                    chars.next();
                }
                tokens.push(Token {
                    kind: TokenKind::And,
                    lexeme: "&".into(),
                });
            }
            '|' => {
                chars.next();
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                tokens.push(Token {
                    kind: TokenKind::Or,
                    lexeme: "|".into(),
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
    UnaryNot(Box<ExprNode>),
    UnaryAbs(Box<ExprNode>),
    UnarySign(Box<ExprNode>),
    UnaryLog(Box<ExprNode>),
    Binary {
        op: BinaryOp,
        left: Box<ExprNode>,
        right: Box<ExprNode>,
    },
    Conditional {
        condition: Box<ExprNode>,
        true_expr: Box<ExprNode>,
        false_expr: Box<ExprNode>,
    },
    // Rolling operators
    RollingMean {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingStd {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingSum {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingVar {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingMax {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingMin {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingQuantile {
        expr: Box<ExprNode>,
        window: usize,
        quantile: f64,
    },
    RollingRank {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingCount {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingSlope {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingRsquare {
        expr: Box<ExprNode>,
        window: usize,
    },
    RollingResi {
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
    Ref {
        expr: Box<ExprNode>,
        periods: usize,
    },
    Delta {
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
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    And,
    Or,
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
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_logical_and()?;
        while self.peek_kind() == Some(TokenKind::Or) {
            self.position += 1;
            let rhs = self.parse_logical_and()?;
            node = ExprNode::Binary {
                op: BinaryOp::Or,
                left: Box::new(node),
                right: Box::new(rhs),
            };
        }
        Ok(node)
    }

    fn parse_logical_and(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_comparison()?;
        while self.peek_kind() == Some(TokenKind::And) {
            self.position += 1;
            let rhs = self.parse_comparison()?;
            node = ExprNode::Binary {
                op: BinaryOp::And,
                left: Box::new(node),
                right: Box::new(rhs),
            };
        }
        Ok(node)
    }

    fn parse_comparison(&mut self) -> Result<ExprNode, OpsError> {
        let mut node = self.parse_add_sub()?;
        loop {
            match self.peek_kind() {
                Some(TokenKind::Greater) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::Less) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Lt,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::GreaterEq) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Ge,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::LessEq) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Le,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::EqEq) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Eq,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                Some(TokenKind::NotEq) => {
                    self.position += 1;
                    let rhs = self.parse_add_sub()?;
                    node = ExprNode::Binary {
                        op: BinaryOp::Ne,
                        left: Box::new(node),
                        right: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
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
        } else if self.match_kind(TokenKind::Not) {
            let expr = self.parse_unary()?;
            Ok(ExprNode::UnaryNot(Box::new(expr)))
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
            // Unary functions
            "abs" => {
                if args.len() != 1 {
                    return Err(OpsError::UnexpectedToken("abs expects one argument".into()));
                }
                Ok(ExprNode::UnaryAbs(Box::new(args[0].clone())))
            }
            "sign" => {
                if args.len() != 1 {
                    return Err(OpsError::UnexpectedToken("sign expects one argument".into()));
                }
                Ok(ExprNode::UnarySign(Box::new(args[0].clone())))
            }
            "log" => {
                if args.len() != 1 {
                    return Err(OpsError::UnexpectedToken("log expects one argument".into()));
                }
                Ok(ExprNode::UnaryLog(Box::new(args[0].clone())))
            }
            // Conditional
            "if" => {
                if args.len() != 3 {
                    return Err(OpsError::UnexpectedToken(
                        "if expects three arguments (condition, true_value, false_value)".into(),
                    ));
                }
                Ok(ExprNode::Conditional {
                    condition: Box::new(args[0].clone()),
                    true_expr: Box::new(args[1].clone()),
                    false_expr: Box::new(args[2].clone()),
                })
            }
            // Rolling mean
            "rolling_mean" | "mean" | "ma" => {
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
            // Rolling std
            "rolling_std" | "std" => {
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
            // Rolling sum
            "rolling_sum" | "sum" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_sum expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingSum {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Rolling var
            "rolling_var" | "var" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_var expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingVar {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Rolling max
            "rolling_max" | "max" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_max expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingMax {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Rolling min
            "rolling_min" | "min" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_min expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingMin {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Rolling quantile
            "rolling_quantile" | "quantile" => {
                if args.len() != 3 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_quantile expects three arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                let quantile = expect_literal_f64(&args[2])?;
                if !(0.0..=1.0).contains(&quantile) {
                    return Err(OpsError::InvalidPercentile);
                }
                Ok(ExprNode::RollingQuantile {
                    expr: Box::new(args[0].clone()),
                    window,
                    quantile,
                })
            }
            // Rolling rank
            "rolling_rank" | "rank" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_rank expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingRank {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Rolling count
            "rolling_count" | "count" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_count expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingCount {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Linear regression operators
            "rolling_slope" | "slope" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_slope expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingSlope {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            "rolling_rsquare" | "rsquare" | "rsqr" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_rsquare expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingRsquare {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            "rolling_resi" | "resi" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "rolling_resi expects two arguments".into(),
                    ));
                }
                let window = expect_literal_usize(&args[1])?;
                if window == 0 {
                    return Err(OpsError::InvalidWindow);
                }
                Ok(ExprNode::RollingResi {
                    expr: Box::new(args[0].clone()),
                    window,
                })
            }
            // Expanding sum
            "expanding_sum" => {
                if args.len() != 1 {
                    return Err(OpsError::UnexpectedToken(
                        "expanding_sum expects one argument".into(),
                    ));
                }
                Ok(ExprNode::ExpandingSum(Box::new(args[0].clone())))
            }
            // Percentile
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
            // Lag and Ref (same operation)
            "lag" | "ref" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        format!("{} expects two arguments", func).as_str().into(),
                    ));
                }
                let periods = expect_literal_usize(&args[1])?;
                if func == "lag" {
                    Ok(ExprNode::Lag {
                        expr: Box::new(args[0].clone()),
                        periods,
                    })
                } else {
                    Ok(ExprNode::Ref {
                        expr: Box::new(args[0].clone()),
                        periods,
                    })
                }
            }
            // Delta
            "delta" => {
                if args.len() != 2 {
                    return Err(OpsError::UnexpectedToken(
                        "delta expects two arguments".into(),
                    ));
                }
                let periods = expect_literal_usize(&args[1])?;
                Ok(ExprNode::Delta {
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
        ExprNode::UnaryNot(expr) => {
            let evaluated = ensure_f64(evaluate(expr, frame)?)?;
            let result: Float64Chunked = evaluated
                .into_iter()
                .map(|value| value.map(|v| if v.abs() < f64::EPSILON { 1.0 } else { 0.0 }))
                .collect();
            Ok(result.into_series())
        }
        ExprNode::UnaryAbs(expr) => {
            let evaluated = ensure_f64(evaluate(expr, frame)?)?;
            let result: Float64Chunked = evaluated
                .into_iter()
                .map(|value| value.map(|v| v.abs()))
                .collect();
            Ok(result.into_series())
        }
        ExprNode::UnarySign(expr) => {
            let evaluated = ensure_f64(evaluate(expr, frame)?)?;
            let result: Float64Chunked = evaluated
                .into_iter()
                .map(|value| value.map(|v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }))
                .collect();
            Ok(result.into_series())
        }
        ExprNode::UnaryLog(expr) => {
            let evaluated = ensure_f64(evaluate(expr, frame)?)?;
            let result: Float64Chunked = evaluated
                .into_iter()
                .map(|value| value.and_then(|v| if v > 0.0 { Some(v.ln()) } else { None }))
                .collect();
            Ok(result.into_series())
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
                    (BinaryOp::Gt, Some(l), Some(r)) => Some(if l > r { 1.0 } else { 0.0 }),
                    (BinaryOp::Lt, Some(l), Some(r)) => Some(if l < r { 1.0 } else { 0.0 }),
                    (BinaryOp::Ge, Some(l), Some(r)) => Some(if l >= r { 1.0 } else { 0.0 }),
                    (BinaryOp::Le, Some(l), Some(r)) => Some(if l <= r { 1.0 } else { 0.0 }),
                    (BinaryOp::Eq, Some(l), Some(r)) => Some(if (l - r).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
                    (BinaryOp::Ne, Some(l), Some(r)) => Some(if (l - r).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
                    (BinaryOp::And, Some(l), Some(r)) => Some(if l.abs() > f64::EPSILON && r.abs() > f64::EPSILON { 1.0 } else { 0.0 }),
                    (BinaryOp::Or, Some(l), Some(r)) => Some(if l.abs() > f64::EPSILON || r.abs() > f64::EPSILON { 1.0 } else { 0.0 }),
                })
                .collect();
            Ok(values.into_series())
        }
        ExprNode::Conditional { condition, true_expr, false_expr } => {
            let cond = ensure_f64(evaluate(condition, frame)?)?;
            let true_vals = ensure_f64(evaluate(true_expr, frame)?)?;
            let false_vals = ensure_f64(evaluate(false_expr, frame)?)?;

            let result: Float64Chunked = cond
                .into_iter()
                .zip(true_vals.into_iter())
                .zip(false_vals.into_iter())
                .map(|((c, t), f)| {
                    match (c, t, f) {
                        (Some(cond), Some(true_val), Some(false_val)) => {
                            if cond.abs() > f64::EPSILON {
                                Some(true_val)
                            } else {
                                Some(false_val)
                            }
                        }
                        _ => None,
                    }
                })
                .collect();
            Ok(result.into_series())
        }
        ExprNode::RollingMean { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_mean(&base, *window).into_series())
        }
        ExprNode::RollingStd { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_std(&base, *window).into_series())
        }
        ExprNode::RollingSum { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_sum(&base, *window).into_series())
        }
        ExprNode::RollingVar { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_var(&base, *window).into_series())
        }
        ExprNode::RollingMax { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_max(&base, *window).into_series())
        }
        ExprNode::RollingMin { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_min(&base, *window).into_series())
        }
        ExprNode::RollingQuantile { expr, window, quantile } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_quantile(&base, *window, *quantile).into_series())
        }
        ExprNode::RollingRank { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_rank(&base, *window).into_series())
        }
        ExprNode::RollingCount { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_count(&base, *window).into_series())
        }
        ExprNode::RollingSlope { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_slope(&base, *window).into_series())
        }
        ExprNode::RollingRsquare { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_rsquare(&base, *window).into_series())
        }
        ExprNode::RollingResi { expr, window } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(rolling_resi(&base, *window).into_series())
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
        ExprNode::Ref { expr, periods } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(lag(&base, *periods).into_series())
        }
        ExprNode::Delta { expr, periods } => {
            let base = ensure_f64(evaluate(expr, frame)?)?;
            Ok(delta(&base, *periods).into_series())
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

// Rolling operators implementations

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

fn rolling_sum(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    let mut sum = 0.0;
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if let Some(v) = value {
                sum += v;
            }
            if queue.len() > window
                && let Some(old) = queue.pop_front().flatten()
            {
                sum -= old;
            }
            Some(sum)
        })
        .collect()
}

fn rolling_var(values: &Float64Chunked, window: usize) -> Float64Chunked {
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
                Some(variance.max(0.0))
            } else {
                Some(0.0)
            }
        })
        .collect()
}

fn rolling_max(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let max = queue.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);
            if max.is_infinite() && max < 0.0 {
                None
            } else {
                Some(max)
            }
        })
        .collect()
}

fn rolling_min(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let min = queue.iter().flatten().copied().fold(f64::INFINITY, f64::min);
            if min.is_infinite() && min > 0.0 {
                None
            } else {
                Some(min)
            }
        })
        .collect()
}

fn rolling_quantile(values: &Float64Chunked, window: usize, quantile: f64) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let mut sorted: Vec<f64> = queue.iter().flatten().copied().collect();
            if sorted.is_empty() {
                return None;
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((sorted.len() - 1) as f64 * quantile).round() as usize;
            Some(sorted[idx.min(sorted.len() - 1)])
        })
        .collect()
}

fn rolling_rank(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            if let Some(current) = value {
                let count_less: usize = queue
                    .iter()
                    .flatten()
                    .filter(|&&v| v < current)
                    .count();
                Some((count_less as f64 + 1.0) / queue.len() as f64)
            } else {
                None
            }
        })
        .collect()
}

fn rolling_count(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            Some(queue.iter().flatten().count() as f64)
        })
        .collect()
}

// Linear regression helpers
fn linear_regression_stats(window_values: &[f64]) -> (f64, f64, f64) {
    if window_values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let n = window_values.len() as f64;
    let x_mean = (1.0 + n) / 2.0;
    let mut y_sum = 0.0;
    let mut xy_sum = 0.0;
    let mut x_sq_sum = 0.0;
    let mut y_sq_sum = 0.0;

    for (i, &y) in window_values.iter().enumerate() {
        let x = (i + 1) as f64;
        y_sum += y;
        xy_sum += x * y;
        x_sq_sum += x * x;
        y_sq_sum += y * y;
    }

    let y_mean = y_sum / n;
    let cov = xy_sum / n - x_mean * y_mean;
    let var_x = x_sq_sum / n - x_mean * x_mean;
    let var_y = y_sq_sum / n - y_mean * y_mean;

    let slope = if var_x > 0.0 { cov / var_x } else { f64::NAN };
    let intercept = y_mean - slope * x_mean;

    let r_square = if var_x > 0.0 && var_y > 0.0 {
        (cov * cov) / (var_x * var_y)
    } else {
        f64::NAN
    };

    let residual = if slope.is_finite() && intercept.is_finite() {
        let last_x = n;
        let last_y = window_values[window_values.len() - 1];
        last_y - (slope * last_x + intercept)
    } else {
        f64::NAN
    };

    (slope, r_square, residual)
}

fn rolling_slope(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let valid: Vec<f64> = queue.iter().flatten().copied().collect();
            if valid.is_empty() {
                return None;
            }
            let (slope, _, _) = linear_regression_stats(&valid);
            Some(slope)
        })
        .collect()
}

fn rolling_rsquare(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let valid: Vec<f64> = queue.iter().flatten().copied().collect();
            if valid.is_empty() {
                return None;
            }
            let (_, r_square, _) = linear_regression_stats(&valid);
            Some(r_square)
        })
        .collect()
}

fn rolling_resi(values: &Float64Chunked, window: usize) -> Float64Chunked {
    let mut queue: VecDeque<Option<f64>> = VecDeque::new();
    values
        .into_iter()
        .map(|value| {
            queue.push_back(value);
            if queue.len() > window {
                queue.pop_front();
            }
            let valid: Vec<f64> = queue.iter().flatten().copied().collect();
            if valid.is_empty() {
                return None;
            }
            let (_, _, residual) = linear_regression_stats(&valid);
            Some(residual)
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

fn delta(values: &Float64Chunked, periods: usize) -> Float64Chunked {
    let lagged = lag(values, periods);
    values
        .into_iter()
        .zip(lagged.into_iter())
        .map(|(current, prev)| {
            match (current, prev) {
                (Some(c), Some(p)) => Some(c - p),
                _ => None,
            }
        })
        .collect()
}
