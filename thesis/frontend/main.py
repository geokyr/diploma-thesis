import atexit

import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from thesis.common.logger import setup_logger
from thesis.common.service import PlatformServiceConfig
from thesis.frontend.callbacks.map import register_map_callbacks
from thesis.frontend.callbacks.monitoring import register_monitoring_callbacks
from thesis.frontend.callbacks.notifications import register_notification_callbacks
from thesis.frontend.callbacks.report import register_report_callbacks
from thesis.frontend.callbacks.simulation import register_simulation_callbacks
from thesis.frontend.layouts.base import create_base_layout
from thesis.frontend.utils.api_client import APIClient

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)
client = APIClient(config.backend_url)

app: dash.Dash = dash.Dash(
    "Platform Frontend",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
)
app.layout = create_base_layout()

load_figure_template("darkly")
register_simulation_callbacks(app, client)
register_monitoring_callbacks(app, client)
register_notification_callbacks(app, client)
register_map_callbacks(app, client)
register_report_callbacks(app, client)


@atexit.register
def shutdown_client() -> None:
    try:
        client.clear()
    except Exception:
        return


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
