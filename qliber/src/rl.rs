use thiserror::Error;

use crate::logging::log_event;

#[derive(Debug, Error)]
pub enum RlError {
    #[error("environment error: {0}")]
    Environment(String),
    #[error("agent error: {0}")]
    Agent(String),
}

pub type RlResult<T> = Result<T, RlError>;

pub trait Environment {
    type State: Clone;
    type Action: Clone;

    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> (Self::State, f64, bool);
}

pub trait Agent<E: Environment> {
    fn act(&mut self, state: &E::State) -> E::Action;
    fn observe(&mut self, state: &E::State, reward: f64, done: bool);
}

pub struct RlTrainer {
    pub episodes: usize,
    pub max_steps: usize,
}

impl Default for RlTrainer {
    fn default() -> Self {
        Self {
            episodes: 10,
            max_steps: 512,
        }
    }
}

impl RlTrainer {
    pub fn new(episodes: usize, max_steps: usize) -> Self {
        Self {
            episodes,
            max_steps,
        }
    }

    pub fn train<E, A>(&self, env: &mut E, agent: &mut A) -> Vec<f64>
    where
        E: Environment,
        A: Agent<E>,
    {
        let mut rewards = Vec::with_capacity(self.episodes);
        for episode in 0..self.episodes {
            let mut state = env.reset();
            let mut total_reward = 0.0;
            for step in 0..self.max_steps {
                let action = agent.act(&state);
                let (next_state, reward, done) = env.step(action);
                total_reward += reward;
                agent.observe(&next_state, reward, done);
                state = next_state;
                if done {
                    break;
                }
                log_event(
                    file!(),
                    "RlTrainer",
                    "train",
                    "rl.step",
                    line!(),
                    &format!("episode={episode} step={step} reward={reward:.4}"),
                    None,
                    "none",
                    "POST",
                );
            }
            rewards.push(total_reward);
        }
        rewards
    }
}

#[derive(Default)]
pub struct CounterEnvironment {
    pub goal: i32,
    pub position: i32,
}

impl Environment for CounterEnvironment {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        self.position = 0;
        self.position
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f64, bool) {
        self.position += action;
        let reward = if self.position >= self.goal {
            1.0
        } else {
            -0.1
        };
        let done = self.position >= self.goal;
        (self.position, reward, done)
    }
}

pub struct IncrementAgent;

impl<E: Environment<State = i32, Action = i32>> Agent<E> for IncrementAgent {
    fn act(&mut self, _: &E::State) -> E::Action {
        1
    }

    fn observe(&mut self, _: &E::State, _: f64, _: bool) {}
}
