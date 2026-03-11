from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from fastchat.serve.gif_arena import (
    DuplicateVoteError,
    GifArenaStore,
    InvalidParticipantIdError,
    normalize_participant_id,
)
from fastchat.serve.gradio_web_server import acknowledgment_md, get_ip
from fastchat.serve.remote_logger import get_remote_logger


def build_gif_arena_ui(
    store: GifArenaStore | None,
    *,
    setup_error: str | None = None,
) -> None:
    notice_markdown = """
# 🎞️ BrowserArena GIF Arena

Vote on archived BrowserArena execution GIFs for existing tasks only. Enter a unique participant ID to receive a task, compare the two archived model attempts for that same task, and record a left/right/tie vote.
"""

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    if setup_error or store is None:
        gr.Markdown(
            f"### GIF arena unavailable\n\n{html.escape(setup_error or 'Unknown configuration error.')}"
        )
        return

    gr.set_static_paths(paths=[store.gif_dir])

    round_state = gr.State()

    with gr.Group(elem_id="gif-arena-region"):
        task_markdown = gr.HTML(_empty_task_markdown())
        with gr.Row():
            with gr.Column():
                left_model_markdown = gr.Markdown("### Model A")
                left_gif_html = gr.HTML(_empty_gif_panel("Model A"))
            with gr.Column():
                right_model_markdown = gr.Markdown("### Model B")
                right_gif_html = gr.HTML(_empty_gif_panel("Model B"))
        status_markdown = gr.Markdown(
            "Enter a participant ID and click `Start Voting` to load the first archived task."
        )

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )

    with gr.Row():
        participant_id_box = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your User ID or MTurk Worker/Prolific ID",
            elem_id="user_id_box",
        )
        next_btn = gr.Button(value="Start Voting", variant="primary")

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    outputs = [
        round_state,
        task_markdown,
        left_model_markdown,
        right_model_markdown,
        left_gif_html,
        right_gif_html,
        status_markdown,
        participant_id_box,
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        next_btn,
    ]

    def render_round(
        round_data: dict[str, object] | None,
        participant_id: str,
        status_text: str,
        *,
        reveal_models: bool,
        voting_enabled: bool,
        next_label: str,
        next_enabled: bool,
    ):
        if round_data is None:
            return (
                None,
                _empty_task_markdown(),
                "### Model A",
                "### Model B",
                _empty_gif_panel("Model A"),
                _empty_gif_panel("Model B"),
                status_text,
                participant_id,
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(visible=False, interactive=False),
                gr.Button(value=next_label, interactive=next_enabled),
            )

        left_attempt = dict(round_data["left_attempt"])
        right_attempt = dict(round_data["right_attempt"])
        left_label = (
            f"### Model A: {left_attempt['model_name']}"
            if reveal_models
            else "### Model A"
        )
        right_label = (
            f"### Model B: {right_attempt['model_name']}"
            if reveal_models
            else "### Model B"
        )

        return (
            round_data,
            _task_markdown(round_data),
            left_label,
            right_label,
            _gif_panel_html(Path(str(round_data["left_gif_path"])), "Model A"),
            _gif_panel_html(Path(str(round_data["right_gif_path"])), "Model B"),
            status_text,
            participant_id,
            gr.Button(value="👈  A is better", visible=True, interactive=voting_enabled),
            gr.Button(value="👉  B is better", visible=True, interactive=voting_enabled),
            gr.Button(value="🤝  Tie", visible=True, interactive=voting_enabled),
            gr.Button(value="👎  Both are bad", visible=True, interactive=voting_enabled),
            gr.Button(value=next_label, interactive=next_enabled),
        )

    def load_next_round(current_round, participant_id_text, request: gr.Request):
        del current_round
        try:
            participant_id = normalize_participant_id(participant_id_text)
        except InvalidParticipantIdError as exc:
            gr.Warning(str(exc))
            return render_round(
                None,
                participant_id_text.strip(),
                str(exc),
                reveal_models=False,
                voting_enabled=False,
                next_label="Start Voting",
                next_enabled=True,
            )

        round_data = store.get_next_round(participant_id)
        if round_data is None:
            return render_round(
                None,
                participant_id,
                "All archived GIF tasks have already been voted on with this participant ID.",
                reveal_models=False,
                voting_enabled=False,
                next_label="All Tasks Completed",
                next_enabled=False,
            )

        status_text = "Review both archived GIFs, then choose the better completion."
        return render_round(
            round_data,
            participant_id,
            status_text,
            reveal_models=False,
            voting_enabled=True,
            next_label="Vote to Continue",
            next_enabled=False,
        )

    def submit_vote(vote_type: str):
        def _submit(round_data, participant_id_text, request: gr.Request):
            if round_data is None:
                gr.Warning("Load an archived GIF task before voting.")
                return render_round(
                    None,
                    participant_id_text.strip(),
                    "Load an archived GIF task before voting.",
                    reveal_models=False,
                    voting_enabled=False,
                    next_label="Start Voting",
                    next_enabled=True,
                )

            try:
                participant_id = normalize_participant_id(participant_id_text)
                summary = store.submit_vote(
                    round_data=round_data,
                    participant_id=participant_id,
                    vote_type=vote_type,
                    requester_ip=get_ip(request),
                )
                get_remote_logger().log(
                    {
                        "type": "gif_vote",
                        "task_id": round_data["task_id"],
                        "source_interaction_id": round_data["source_interaction_id"],
                        "participant_id": participant_id,
                        "vote_type": vote_type,
                        "left_model": round_data["left_attempt"]["model_name"],
                        "right_model": round_data["right_attempt"]["model_name"],
                        "left_gif_id": round_data["left_attempt"]["gif_id"],
                        "right_gif_id": round_data["right_attempt"]["gif_id"],
                        "ip": get_ip(request),
                    }
                )
                gr.Info("Thanks for voting.")
                next_label = (
                    "Next Task"
                    if summary["remaining_tasks"] > 0
                    else "All Tasks Completed"
                )
                return render_round(
                    round_data,
                    participant_id,
                    "Vote recorded.",
                    reveal_models=True,
                    voting_enabled=False,
                    next_label=next_label,
                    next_enabled=summary["remaining_tasks"] > 0,
                )
            except DuplicateVoteError as exc:
                gr.Warning(str(exc))
                return render_round(
                    round_data,
                    participant_id_text.strip(),
                    str(exc),
                    reveal_models=True,
                    voting_enabled=False,
                    next_label="Next Task",
                    next_enabled=True,
                )
            except InvalidParticipantIdError as exc:
                gr.Warning(str(exc))
                return render_round(
                    round_data,
                    participant_id_text.strip(),
                    str(exc),
                    reveal_models=False,
                    voting_enabled=True,
                    next_label="Vote to Continue",
                    next_enabled=False,
                )

        return _submit

    next_btn.click(
        load_next_round,
        [round_state, participant_id_box],
        outputs,
    )
    leftvote_btn.click(
        submit_vote("leftvote"),
        [round_state, participant_id_box],
        outputs,
    )
    rightvote_btn.click(
        submit_vote("rightvote"),
        [round_state, participant_id_box],
        outputs,
    )
    tie_btn.click(
        submit_vote("tievote"),
        [round_state, participant_id_box],
        outputs,
    )
    bothbad_btn.click(
        submit_vote("bothbad_vote"),
        [round_state, participant_id_box],
        outputs,
    )


def _empty_task_markdown() -> str:
    return (
        "<div style='border:2px solid #d8d8d8;border-radius:14px;padding:20px;"
        "background:linear-gradient(180deg,#fffdf7 0%,#f5f1e8 100%);'>"
        "<div style='font-size:13px;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#7a6a3a;margin-bottom:10px;'>"
        "Archived BrowserArena Task"
        "</div>"
        "<div style='font-size:26px;line-height:1.35;font-weight:700;color:#1f1f1f;'>"
        "The next side-by-side GIF comparison will appear here once a participant ID is provided."
        "</div>"
        "</div>"
    )


def _task_markdown(round_data: dict[str, object]) -> str:
    return (
        "<div style='border:2px solid #d7c48a;border-radius:14px;padding:20px;"
        "background:linear-gradient(180deg,#fffaf0 0%,#f4ecd6 100%);"
        "box-shadow:0 10px 24px rgba(0,0,0,0.05);'>"
        f"<div style='font-size:13px;font-weight:700;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:#7a5d12;margin-bottom:10px;'>"
        f"Task {round_data['source_interaction_id']}"
        "</div>"
        f"<div style='font-size:28px;line-height:1.35;font-weight:700;color:#171717;'>"
        f"{html.escape(str(round_data['task_text']))}"
        "</div>"
        "</div>"
    )


def _empty_gif_panel(label: str) -> str:
    return (
        f"<div style='min-height:420px;border:1px solid #ddd;border-radius:12px;"
        f"padding:24px;background:#f7f7f7;display:flex;align-items:center;"
        f"justify-content:center;color:#666;'>"
        f"{html.escape(label)} GIF will appear here."
        f"</div>"
    )


def _gif_panel_html(gif_path: Path, label: str) -> str:
    escaped_path = html.escape(gif_path.as_posix(), quote=True)
    escaped_label = html.escape(label)
    return (
        "<div style='border:1px solid #ddd;border-radius:12px;padding:12px;"
        "background:#f7f7f7;'>"
        f"<img src='/gradio_api/file={escaped_path}' alt='{escaped_label}' "
        "style='width:100%;max-height:540px;object-fit:contain;border-radius:8px;"
        "background:#fff;' />"
        "<div style='margin-top:10px;font-size:14px;color:#666;'>Archived execution GIF</div>"
        "</div>"
    )
