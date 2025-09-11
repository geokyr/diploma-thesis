import requests

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS, MAX_INTERVALS, SPEED_MULTIPLIER


# TODO: tighten dict types
class ApiClient:
    def __init__(self, backend_url: str, timeout: float = HTTP_CLIENT_TIMEOUT_SECONDS) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(f"{self.backend_url}{path}", params=params, timeout=self.timeout)
        return resp.json() if resp.ok else {"status": "error"}

    def _post(self, path: str, json_body: dict | None = None) -> dict:
        resp = self.session.post(f"{self.backend_url}{path}", json=json_body, timeout=self.timeout)
        return resp.json() if resp.ok else {"status": "error"}

    def fetch_history(self, limit: int = MAX_INTERVALS) -> dict:
        return self._get("/metrics/history", params={"limit": limit})

    def simulation_start(self, speed_multiplier: float = SPEED_MULTIPLIER) -> dict:
        return self._post("/simulation/start", {"speed_multiplier": speed_multiplier})

    def simulation_pause(self) -> dict:
        return self._post("/simulation/pause")

    def simulation_resume(self) -> dict:
        return self._post("/simulation/resume")

    def simulation_reset(self) -> dict:
        return self._post("/simulation/reset")

    def fetch_status(self) -> dict:
        return self._get("/simulation/status")
