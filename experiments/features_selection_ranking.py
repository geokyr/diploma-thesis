from thesis.common.enums import ETAEvaluation
from thesis.common.logger import setup_logger
from thesis.eta.experiment import ETAExperiment
from thesis.eta.features import (
    add_feature_categories,
    combine_feature_rankings,
    load_feature_selection_results,
    save_feature_selection_results,
)


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    feature_selection_results = load_feature_selection_results()
    final_ranking = combine_feature_rankings(feature_selection_results)
    final_ranking_with_categories = add_feature_categories(final_ranking)
    save_feature_selection_results(final_ranking_with_categories, experiment.results_dir)


if __name__ == "__main__":
    main()
