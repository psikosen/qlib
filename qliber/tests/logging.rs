use qliber::logging;

#[test]
fn logging_initialization_is_idempotent() {
    logging::init_logging().expect("first initialization succeeds");
    logging::init_logging().expect("subsequent initialization succeeds");
}
