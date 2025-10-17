from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from thesis.common.enums import PlatformService, PlatformServiceStatus
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformServiceConfig
from thesis.summarizer.routers.report import report_router
from thesis.summarizer.services.report_generator import ReportGenerator

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        report_generator = ReportGenerator()

        app.state.report_generator = report_generator
        yield
    finally:
        report_generator: ReportGenerator = getattr(app.state, "report_generator", None)

        if report_generator is not None:
            await report_generator.clear()

        if hasattr(app.state, "report_generator"):
            delattr(app.state, "report_generator")


app = FastAPI(
    title="Platform Summarizer Service", version="1.0.0", lifespan=lifespan, default_response_class=ORJSONResponse
)
app.include_router(report_router, prefix="/report", tags=["report"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    return HealthResponse(status=PlatformServiceStatus.HEALTHY, service=PlatformService.SUMMARIZER)


if __name__ == "__main__":
    uvicorn.run(
        "thesis.summarizer.main:app",
        host=config.host,
        port=config.port,
        reload=config.is_development,
        loop=config.loop,
        http=config.http,
        access_log=config.access_log,
    )
