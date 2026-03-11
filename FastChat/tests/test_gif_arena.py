import pytest

from fastchat.serve.gif_arena import (
    DEFAULT_SURVEY_PATH,
    InvalidParticipantIdError,
    compute_elo_ratings,
    normalize_participant_id,
    parse_survey_interactions,
    select_relevant_attempt_pair,
)


def test_parse_survey_interactions_extracts_verified_subset():
    interactions = parse_survey_interactions(DEFAULT_SURVEY_PATH)

    assert len(interactions) == 25
    assert sum(len(interaction.attempts) for interaction in interactions) == 50
    assert interactions[0].task_text.startswith(
        "How many nonstop flights to Las Vegas depart from Portland"
    )
    assert interactions[0].attempts[0].gif_id == "12_05_2025_03_54_21_KhV.gif"


def test_compute_elo_ratings_uses_archived_votes():
    interactions = parse_survey_interactions(DEFAULT_SURVEY_PATH)
    vote_rows = [
        (
            interaction.attempts[0].model_name,
            interaction.attempts[1].model_name,
            interaction.source_vote_type,
        )
        for interaction in interactions
    ]

    ratings = compute_elo_ratings(vote_rows)

    assert len(ratings) == 5
    assert min(ratings, key=ratings.get) == "google/gemini-2.5-pro-preview-03-25"
    assert max(ratings, key=ratings.get) == "meta-llama/llama-4-maverick"


def test_select_relevant_attempt_pair_prefers_smallest_elo_gap():
    attempts = [
        {
            "attempt_id": 11,
            "source_attempt_index": 1,
            "model_name": "model-a",
            "gif_id": "a.gif",
            "log_id": "a",
            "status": "success",
        },
        {
            "attempt_id": 12,
            "source_attempt_index": 2,
            "model_name": "model-b",
            "gif_id": "b.gif",
            "log_id": "b",
            "status": "success",
        },
        {
            "attempt_id": 13,
            "source_attempt_index": 3,
            "model_name": "model-c",
            "gif_id": "c.gif",
            "log_id": "c",
            "status": "success",
        },
    ]
    ratings = {"model-a": 1000.0, "model-b": 1012.0, "model-c": 1310.0}

    left_attempt, right_attempt, gap, pair_vote_count = select_relevant_attempt_pair(
        attempts,
        ratings,
    )

    assert {left_attempt["attempt_id"], right_attempt["attempt_id"]} == {11, 12}
    assert gap == pytest.approx(12.0)
    assert pair_vote_count == 0


def test_normalize_participant_id_rejects_blank_values():
    with pytest.raises(InvalidParticipantIdError):
        normalize_participant_id("   ")

    assert normalize_participant_id(" worker-123 ") == "worker-123"
