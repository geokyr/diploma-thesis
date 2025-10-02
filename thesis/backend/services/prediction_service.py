"""Prediction service for handling user trip predictions."""

import httpx

from thesis.backend.services.timelapse_driver import TimelapseDriver
from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.enums import MLTask
from thesis.common.schemas import (
    PredictionSingleRequest,
    PredictionSingleResponse,
    TripPredictionResponse,
)


class PredictionService:
    """Prediction service for handling user trip predictions."""

    def __init__(self, timelapse_driver: TimelapseDriver) -> None:
        self._timelapse_driver = timelapse_driver
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS)

    async def predict_trip(self, prediction_request: PredictionSingleRequest) -> TripPredictionResponse:
        """
        Predict trip metrics for all available ML tasks.

        Args:
            prediction_request (PredictionSingleRequest): Trip prediction request.

        Returns:
            TripPredictionResponse: Predictions per ML task.
        """
        available_tasks = self._timelapse_driver.ml_tasks
        predictions: dict[MLTask, PredictionSingleResponse] = {}

        for ml_task in available_tasks:
            result = await self._predict_single_task(ml_task, prediction_request)
            if result.prediction is not None:
                predictions[ml_task] = result

        return TripPredictionResponse(predictions=predictions)

    async def _predict_single_task(
        self, ml_task: MLTask, prediction_request: PredictionSingleRequest
    ) -> PredictionSingleResponse:
        """
        Predict for a single ML task.

        Args:
            ml_task (MLTask): The ML task to predict for.
            prediction_request (PredictionSingleRequest): Trip details.

        Returns:
            PredictionSingleResponse: Prediction value.
        """
        url = f"{self._timelapse_driver.predictor_urls[ml_task]}/predict/single"
        payload = prediction_request.model_dump()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()

            return PredictionSingleResponse.model_validate(response.json())

        except Exception:
            return PredictionSingleResponse(prediction=None)

    async def clear(self) -> None:
        """Clear the prediction service."""
        await self._client.aclose()
