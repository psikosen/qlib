use qliber::rl::{CounterEnvironment, IncrementAgent, RlTrainer};

#[test]
fn rl_trainer_accumulates_rewards() {
    let mut env = CounterEnvironment {
        goal: 3,
        position: 0,
    };
    let mut agent = IncrementAgent;
    let trainer = RlTrainer::new(1, 10);
    let rewards = trainer.train(&mut env, &mut agent);
    assert_eq!(rewards.len(), 1);
    assert!(rewards[0] > 0.0);
}
