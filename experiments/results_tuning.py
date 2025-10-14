from thesis.common.enums import ETAEvaluation
from thesis.common.logger import setup_logger
from thesis.eta.experiment import ETAExperiment
from thesis.eta.results import (
    load_all_tuning_results,
    plot_parameters_vs_mae,
    run_tuning_analysis,
    save_all_tuning_results,
)


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    results = load_all_tuning_results()
    save_all_tuning_results(results, experiment.results_dir)

    run_tuning_analysis(results, experiment.results_dir)
    plot_parameters_vs_mae(results, experiment.results_dir)


if __name__ == "__main__":
    main()
