"""Callbacks for AI summary report functionality."""

import io

import dash_bootstrap_components as dbc
import markdown
from dash import Input, Output, State, ctx, dash, dcc, html, no_update
from xhtml2pdf import pisa

from thesis.common.config import CSS, SIMULATION_REPORT_FILENAME
from thesis.common.enums import MLTask, ReportStatus, SimulationState
from thesis.common.schemas import SimulationSnapshot
from thesis.frontend.utils.api_client import APIClient
from thesis.frontend.utils.formatting import get_report_status_text


def register_report_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all summary report-related callbacks.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("report-interval", "disabled"),
        Input("snapshot-store", "data"),
        Input("report-store", "data"),
    )
    def manage_report_interval(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, dict]],
        report_data: dict[str, str | None],
    ) -> bool:
        """
        Enable report interval when simulation is completed and report is not ready/failed.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, dict]]): Snapshot data.
            report_data (dict[str, str | None]): Report state with status and content.

        Returns:
            bool: Whether the report interval should be disabled.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)
            report_status = ReportStatus(report_data.get("status", ReportStatus.NOT_STARTED.value))

            is_completed = snapshot.state == SimulationState.COMPLETED
            report_in_progress = report_status in (ReportStatus.NOT_STARTED, ReportStatus.GENERATING)

            return not (is_completed and report_in_progress)
        except Exception:
            return True

    @app.callback(
        Output("report-store", "data"),
        Input("report-interval", "n_intervals"),
    )
    def poll_report_status(
        n_intervals: int,
    ) -> dict[str, str | None]:
        """
        Poll backend for report status and content.

        Args:
            n_intervals (int): Number of intervals triggered.

        Returns:
            dict[str, str | None]: Updated report state.
        """
        try:
            report_response = client.simulation_report()
            return report_response.model_dump(mode="json")
        except Exception:
            return no_update

    @app.callback(
        Output("ai-report-view-button", "disabled"),
        Output("download-pdf-button", "disabled"),
        Output("ai-report-status-text", "children"),
        Input("report-store", "data"),
        prevent_initial_call=False,
    )
    def update_button_states(report_data: dict[str, str | None]) -> tuple[bool, bool, str]:
        """
        Update button states and status text based on report status.

        Args:
            report_data (dict[str, str | None]): Report state with status and content.

        Returns:
            tuple[bool, bool, str]: View button disabled, Download button disabled, and status text.
        """
        report_status = report_data["status"]
        buttons_disabled = report_status != ReportStatus.READY
        status_text = get_report_status_text(report_status)

        return buttons_disabled, buttons_disabled, status_text

    @app.callback(
        Output("report-modal", "is_open"),
        Input("ai-report-view-button", "n_clicks"),
        State("report-store", "data"),
        prevent_initial_call=True,
    )
    def open_report_modal(view_clicks: int | None, report_data: dict[str, str | None]) -> bool:
        """
        Open report modal when view button is clicked.

        Args:
            view_clicks (int | None): Number of times view button was clicked.
            report_data (dict[str, str | None]): Current report state.

        Returns:
            bool: Modal open state.
        """
        if not ctx.triggered or not view_clicks:
            return no_update

        stored_content = report_data["content"]

        if stored_content is None:
            return False

        return True

    @app.callback(
        Output("report-modal-body", "children"),
        Input("report-store", "data"),
    )
    def render_report_content(report_data: dict[str, str | None]) -> list | str:
        """
        Render markdown report content as HTML.

        Args:
            report_data (dict[str, str | None]): Report state with status and content.

        Returns:
            list | str: Rendered HTML content or message.
        """
        content = report_data["content"]

        if content is None:
            return html.Div(dbc.Spinner(color="info"), className="d-flex justify-content-center align-items-center")

        try:
            return [dcc.Markdown(content, dangerously_allow_html=True)]
        except Exception:
            return "Error rendering report content."

    @app.callback(
        Output("download-report-pdf", "data"),
        Input("download-pdf-button", "n_clicks"),
        State("report-store", "data"),
        prevent_initial_call=True,
    )
    def download_pdf_report(n_clicks: int | None, report_data: dict[str, str | None]) -> dict[str, bytes]:
        """
        Download the report as PDF, generated on frontend.

        Args:
            n_clicks (int | None): Number of times button was clicked.
            report_data (dict[str, str | None]): Report state with status and content.

        Returns:
            dict[str, bytes]: Download data for dcc.Download component.
        """
        markdown_content = report_data["content"]

        if n_clicks and markdown_content:
            try:
                html_content = markdown.markdown(markdown_content, extensions=["sane_lists"])
                full_html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_content}</body></html>"

                buffer = io.BytesIO()
                status = pisa.CreatePDF(src=io.StringIO(full_html), dest=buffer, encoding="utf-8")

                if status.err:
                    return no_update

                buffer.seek(0)
                return dcc.send_bytes(buffer.read(), SIMULATION_REPORT_FILENAME)
            except Exception:
                return no_update

        return no_update
