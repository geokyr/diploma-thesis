from thesis.common.logger import setup_logger
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.results import load_research_results, run_metric_analysis, save_research_results


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    results_df = load_research_results()
    save_research_results(results_df, experiment.results_dir)

    metrics = [
        ("mae", "MAE", "s", 1),
        ("mape", "MAPE", "%", 1),
        ("rmse", "RMSE", "s", 1),
        ("training_time", "Training Time", "s", 1),
    ]

    for metric_column, metric_name, metric_unit, decimal_places in metrics:
        run_metric_analysis(results_df, metric_column, metric_name, metric_unit, decimal_places, experiment.results_dir)


if __name__ == "__main__":
    main()
