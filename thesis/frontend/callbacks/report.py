"""Callbacks for AI summary report functionality."""

import io
import re

import dash_bootstrap_components as dbc
import markdown
from dash import Input, Output, State, ctx, dash, dcc, html, no_update
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from thesis.common.config import SIMULATION_REPORT_FILENAME
from thesis.common.enums import MLTask, NotificationLevel, ReportStatus
from thesis.frontend.utils.api_client import APIClient


def register_report_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all summary report-related callbacks.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("report-store", "data"),
        Input("notifications-store", "data"),
        Input("simulation-interval", "n_intervals"),
    )
    def poll_report_status(
        notifications_data: list[dict[str, int | str | NotificationLevel | MLTask | None]],
        n_intervals: int,
    ) -> dict[str, str | None]:
        """
        Poll backend for report status and content after simulation ends.

        Args:
            notifications_data (list[dict[str, int | str | NotificationLevel | MLTask | None]]): Notifications data.
            n_intervals (int): Number of intervals triggered.

        Returns:
            dict[str, str | None]: Updated report state.
        """
        if not notifications_data:
            return no_update

        data_exhausted = any("exhausted" in notification["message"].lower() for notification in notifications_data)

        if not data_exhausted:
            return no_update

        try:
            report_response = client.simulation_report()
            return report_response.model_dump(mode="json")
        except Exception:
            return no_update

    @app.callback(
        Output("ai-report-button", "disabled"),
        Output("ai-report-tooltip-container", "children"),
        Input("report-store", "data"),
        Input("bootstrap-interval", "n_intervals"),
    )
    def update_button_state(report_data: dict[str, str | None], n_intervals: int) -> tuple[bool, dbc.Tooltip | None]:
        """
        Update button state and tooltip based on report status.

        Args:
            report_data (dict[str, str | None]): Report state with status and content.
            n_intervals (int): Bootstrap interval count for initial trigger.

        Returns:
            tuple[bool, dbc.Tooltip | None]: Button disabled state and tooltip component.
        """
        report_status = report_data["status"]
        button_disabled = report_status != ReportStatus.READY

        tooltip = (
            dbc.Tooltip(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "AI SummaryReport will be available after simulation completes",
                ],
                target="ai-report-button-container",
                placement="top",
            )
            if button_disabled
            else None
        )

        return button_disabled, tooltip

    @app.callback(
        Output("report-modal", "is_open"),
        Output("report-store", "data", allow_duplicate=True),
        Output("download-pdf-button", "disabled"),
        Input("ai-report-button", "n_clicks"),
        State("report-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def open_report_modal(
        open_clicks: int | None, report_data: dict[str, str | None]
    ) -> tuple[bool, dict[str, str | None], bool]:
        """
        Open report modal and fetch report content.

        Args:
            open_clicks (int | None): Number of times open button was clicked.
            report_data (dict[str, str | None]): Current report state.

        Returns:
            tuple[bool, dict[str, str | None], bool]: Modal open state, updated report data, and PDF button disabled state.
        """
        if not ctx.triggered or not open_clicks:
            return no_update, no_update, no_update

        stored_content = report_data["content"]

        if stored_content is None:
            return False, no_update, True

        return True, no_update, False

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
            return html.Div(dbc.Spinner(color="primary"), className="d-flex justify-content-center align-items-center")

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
                buffer = io.BytesIO()

                html_content = markdown.markdown(
                    markdown_content, extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
                )

                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                lines = html_content.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 0.2 * inch))
                        continue

                    text = re.sub(r"<[^>]+>", "", line)
                    if not text:
                        continue

                    if line.startswith("<h1>"):
                        story.append(Paragraph(text, styles["Heading1"]))
                    elif line.startswith("<h2>"):
                        story.append(Paragraph(text, styles["Heading2"]))
                    elif line.startswith("<h3>"):
                        story.append(Paragraph(text, styles["Heading3"]))
                    elif line.startswith("<h4>"):
                        story.append(Paragraph(text, styles["Heading4"]))
                    elif line.startswith("<h5>"):
                        story.append(Paragraph(text, styles["Heading5"]))
                    elif line.startswith("<h6>"):
                        story.append(Paragraph(text, styles["Heading6"]))
                    elif line.startswith("<code>") or line.startswith("<pre>"):
                        story.append(Paragraph(text, styles["Code"]))
                    elif line.startswith("<p>"):
                        story.append(Paragraph(text, styles["BodyText"]))
                    elif line.startswith("<li>"):
                        story.append(Paragraph(f"• {text}", styles["BodyText"]))
                    elif line.startswith("<strong>") or line.startswith("<b>"):
                        story.append(Paragraph(f"<b>{text}</b>", styles["BodyText"]))
                    elif line.startswith("<em>") or line.startswith("<i>"):
                        story.append(Paragraph(f"<i>{text}</i>", styles["BodyText"]))
                    elif not line.startswith("</"):
                        story.append(Paragraph(text, styles["BodyText"]))

                doc.build(story)
                buffer.seek(0)

                return dcc.send_bytes(buffer.read(), SIMULATION_REPORT_FILENAME)
            except Exception:
                return no_update

        return no_update
