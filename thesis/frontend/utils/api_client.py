import requests


# TODO: add defaults to config
class ApiClient:
    def __init__(self, backend_url: str, timeout: float = 2.0) -> None:
        self.backend_url = backend_url
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(f"{self.backend_url}{path}", params=params, timeout=self.timeout)
        return resp.json() if resp.ok else {}

    def _post(self, path: str, json_body: dict | None = None) -> dict:
        resp = self.session.post(f"{self.backend_url}{path}", json=json_body, timeout=self.timeout)
        return resp.json() if resp.ok else {"status": "error"}

    def fetch_history(self, limit: int = 240) -> dict:
        return self._get("/metrics/history", params={"limit": limit})

    def simulation_start(self, speed_multiplier: float = 300.0) -> dict:
        return self._post("/simulation/start", {"speed_multiplier": speed_multiplier})

    def simulation_pause(self) -> dict:
        return self._post("/simulation/pause")

    def simulation_resume(self) -> dict:
        return self._post("/simulation/resume")

    def simulation_reset(self) -> dict:
        return self._post("/simulation/reset")

    def fetch_status(self) -> dict:
        return self._get("/simulation/status")
