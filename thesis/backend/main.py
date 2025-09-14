import uvicorn
from fastapi import FastAPI

from thesis.common.logger import setup_logger
from thesis.common.service import PlatformService, PlatformServiceConfig

from .routers.metrics import router as metrics_router
from .routers.simulation import router as simulation_router

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)

app = FastAPI(title="Platform Backend API", version="1.0.0")


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy", "service": PlatformService.BACKEND}


app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

if __name__ == "__main__":
    uvicorn.run("thesis.backend.main:app", host=config.host, port=config.port, reload=config.is_development)
