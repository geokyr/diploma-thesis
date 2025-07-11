from thesis.common.logger import setup_logger
from thesis.eta.experiment import initialize_experiment


def main() -> None:
    experiment_name = "comparison"
    _, logs_dir, _, _ = initialize_experiment(experiment_name)
    logger = setup_logger(experiment_name, logs_dir)


if __name__ == "__main__":
    main()
