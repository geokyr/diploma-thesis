from fastapi import APIRouter

from thesis.common.enums import DriftState
from thesis.common.schemas import DriftErrorsRequest

router = APIRouter()


@router.post("/errors")
def process_drift_errors(req: DriftErrorsRequest):
    # TODO: forward to actual drift detection pipeline
    return {"task": req.task, "state": DriftState.STABLE, "start_timestamp": 0.0}
