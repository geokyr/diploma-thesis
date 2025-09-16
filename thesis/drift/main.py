import uvicorn
from fastapi import FastAPI

from thesis.common.logger import setup_logger
from thesis.common.service import PlatformService, PlatformServiceConfig

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)

app = FastAPI(title="Platform Drift Service", version="1.0.0")


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy", "service": PlatformService.DRIFT}


if __name__ == "__main__":
    uvicorn.run("thesis.drift.main:app", host=config.host, port=config.port, reload=config.is_development)
