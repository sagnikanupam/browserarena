from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import math
import os
import random
import re
from typing import Iterable, Mapping, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SURVEY_PATH = REPO_ROOT / "verified-gifs-only" / "survey_interactions.txt"
DEFAULT_GIF_DIR = REPO_ROOT / "verified-gifs-only" / "gifs"
DEFAULT_DB_DSN = os.environ.get("BROWSERARENA_DB_DSN") or os.environ.get(
    "DATABASE_URL"
) or "dbname=browserarena"
SEED_PARTICIPANT_ID = "seed-survey"
ALLOWED_VOTE_TYPES = {"leftvote", "rightvote", "tievote", "bothbad_vote"}
INTERACTION_SEPARATOR = "=" * 70
INTERACTION_HEADER_RE = re.compile(
    r"INTERACTION (\d+)\n"
    r"Date/Time: (.+)\n"
    r"Vote Type: (.+)\n"
    r"User Task: (.+)",
    re.MULTILINE,
)
ATTEMPT_RE = re.compile(
    r"^📱 MODEL ATTEMPT (\d+): (.+)\n"
    r"  Model: (.+)\n"
    r"  GIF ID: (.+)\n"
    r"  Log ID: (.+)$",
    re.MULTILINE,
)


class GifArenaError(RuntimeError):
    pass


class InvalidParticipantIdError(GifArenaError):
    pass


class DuplicateVoteError(GifArenaError):
    pass


@dataclass(frozen=True)
class GifArenaAttempt:
    source_attempt_index: int
    model_name: str
    gif_id: str
    log_id: str
    status: str


@dataclass(frozen=True)
class GifArenaInteraction:
    source_interaction_id: int
    recorded_at: datetime | None
    source_vote_type: str
    task_text: str
    task_fingerprint: str
    attempts: tuple[GifArenaAttempt, ...]


def normalize_participant_id(participant_id: str) -> str:
    normalized = participant_id.strip()
    if not normalized:
        raise InvalidParticipantIdError("Participant ID is required.")
    if len(normalized) > 128:
        raise InvalidParticipantIdError("Participant ID must be 128 characters or fewer.")
    return normalized


def parse_survey_interactions(survey_path: str | Path) -> list[GifArenaInteraction]:
    survey_path = Path(survey_path)
    if not survey_path.exists():
        raise GifArenaError(f"GIF arena survey file was not found: {survey_path}")

    text = survey_path.read_text(encoding="utf-8")
    chunks = text.split(f"{INTERACTION_SEPARATOR}\nINTERACTION ")
    interactions: list[GifArenaInteraction] = []

    for index, raw_chunk in enumerate(chunks):
        if index == 0:
            continue
        chunk = "INTERACTION " + raw_chunk
        header_match = INTERACTION_HEADER_RE.search(chunk)
        if header_match is None:
            continue

        source_interaction_id = int(header_match.group(1))
        recorded_at = _parse_recorded_at(header_match.group(2).strip())
        source_vote_type = header_match.group(3).strip()
        task_text = header_match.group(4).strip()
        if source_vote_type not in ALLOWED_VOTE_TYPES:
            raise GifArenaError(
                f"Unexpected source vote type {source_vote_type!r} in interaction "
                f"{source_interaction_id}."
            )

        attempts = tuple(
            GifArenaAttempt(
                source_attempt_index=int(match.group(1)),
                status=match.group(2).strip(),
                model_name=match.group(3).strip(),
                gif_id=match.group(4).strip(),
                log_id=match.group(5).strip(),
            )
            for match in ATTEMPT_RE.finditer(chunk)
        )
        if len(attempts) < 2:
            raise GifArenaError(
                f"Expected at least two attempts for interaction {source_interaction_id}, "
                f"found {len(attempts)}."
            )

        interactions.append(
            GifArenaInteraction(
                source_interaction_id=source_interaction_id,
                recorded_at=recorded_at,
                source_vote_type=source_vote_type,
                task_text=task_text,
                task_fingerprint=_build_task_fingerprint(task_text, attempts),
                attempts=attempts,
            )
        )

    if not interactions:
        raise GifArenaError(f"No survey interactions were parsed from {survey_path}.")
    return interactions


def compute_elo_ratings(
    vote_rows: Iterable[tuple[str, str, str]],
    *,
    initial_rating: float = 1000.0,
    k_factor: float = 32.0,
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    for model_a, model_b, vote_type in vote_rows:
        rating_a = ratings.get(model_a, initial_rating)
        rating_b = ratings.get(model_b, initial_rating)
        expected_a = 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a

        if vote_type == "leftvote":
            score_a, score_b = 1.0, 0.0
        elif vote_type == "rightvote":
            score_a, score_b = 0.0, 1.0
        else:
            score_a = score_b = 0.5

        ratings[model_a] = rating_a + k_factor * (score_a - expected_a)
        ratings[model_b] = rating_b + k_factor * (score_b - expected_b)

    return ratings


def select_relevant_attempt_pair(
    attempts: Sequence[Mapping[str, object]],
    model_ratings: Mapping[str, float],
    pair_vote_counts: Mapping[tuple[int, int], int] | None = None,
) -> tuple[dict[str, object], dict[str, object], float, int]:
    if len(attempts) < 2:
        raise GifArenaError("A GIF arena round requires at least two attempts.")

    pair_vote_counts = pair_vote_counts or {}
    best_pair: tuple[dict[str, object], dict[str, object]] | None = None
    best_gap = float("inf")
    best_pair_vote_count = 0

    for left_index in range(len(attempts) - 1):
        for right_index in range(left_index + 1, len(attempts)):
            left_attempt = dict(attempts[left_index])
            right_attempt = dict(attempts[right_index])
            left_rating = model_ratings.get(str(left_attempt["model_name"]), 1000.0)
            right_rating = model_ratings.get(str(right_attempt["model_name"]), 1000.0)
            gap = abs(left_rating - right_rating)

            pair_key = tuple(
                sorted(
                    (
                        int(left_attempt["attempt_id"]),
                        int(right_attempt["attempt_id"]),
                    )
                )
            )
            pair_vote_count = pair_vote_counts.get(pair_key, 0)
            candidate_priority = (
                gap,
                pair_vote_count,
                int(left_attempt["source_attempt_index"]),
                int(right_attempt["source_attempt_index"]),
            )
            best_priority = (
                best_gap,
                best_pair_vote_count,
                int(best_pair[0]["source_attempt_index"]) if best_pair else math.inf,
                int(best_pair[1]["source_attempt_index"]) if best_pair else math.inf,
            )
            if best_pair is None or candidate_priority < best_priority:
                best_pair = (left_attempt, right_attempt)
                best_gap = gap
                best_pair_vote_count = pair_vote_count

    assert best_pair is not None
    return best_pair[0], best_pair[1], best_gap, best_pair_vote_count


class GifArenaStore:
    def __init__(
        self,
        *,
        survey_path: str | Path = DEFAULT_SURVEY_PATH,
        gif_dir: str | Path = DEFAULT_GIF_DIR,
        dsn: str | None = None,
    ) -> None:
        self.survey_path = Path(survey_path).resolve()
        self.gif_dir = Path(gif_dir).resolve()
        self.dsn = dsn or DEFAULT_DB_DSN
        self.interactions = parse_survey_interactions(self.survey_path)
        self.unique_task_fingerprints = {
            interaction.task_fingerprint for interaction in self.interactions
        }

        self._validate_gif_files()
        try:
            self.pool = ThreadedConnectionPool(1, 8, self.dsn)
        except psycopg2.OperationalError as exc:
            raise GifArenaError(
                "Unable to connect to the GIF arena Postgres database. "
                "Set BROWSERARENA_DB_DSN or DATABASE_URL, or create the default "
                "database named 'browserarena'."
            ) from exc

        self._install_schema()
        self._bootstrap_seed_data()

    @contextmanager
    def connection(self):
        connection = self.pool.getconn()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self.pool.putconn(connection)

    def get_next_round(self, participant_id: str) -> dict[str, object] | None:
        participant_id = normalize_participant_id(participant_id)
        with self.connection() as connection, connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:
            participant_votes = self._get_participant_vote_count(cursor, participant_id)

            cursor.execute(
                """
                SELECT
                    tasks.id AS task_id,
                    tasks.source_interaction_id,
                    tasks.task_text,
                    COALESCE(task_vote_counts.vote_count, 0) AS task_vote_count
                FROM gif_arena_tasks AS tasks
                LEFT JOIN (
                    SELECT task_id, COUNT(*) FILTER (WHERE NOT is_seed) AS vote_count
                    FROM gif_arena_votes
                    GROUP BY task_id
                ) AS task_vote_counts
                    ON task_vote_counts.task_id = tasks.id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM gif_arena_votes AS participant_votes
                    WHERE participant_votes.task_id = tasks.id
                        AND participant_votes.participant_id = %s
                        AND NOT participant_votes.is_seed
                )
                ORDER BY tasks.source_interaction_id ASC
                """,
                (participant_id,),
            )
            task_rows = cursor.fetchall()
            if not task_rows:
                return None

            task_ids = [int(row["task_id"]) for row in task_rows]
            cursor.execute(
                """
                SELECT
                    id AS attempt_id,
                    task_id,
                    source_attempt_index,
                    model_name,
                    gif_id,
                    log_id,
                    status
                FROM gif_arena_attempts
                WHERE task_id = ANY(%s)
                ORDER BY task_id ASC, source_attempt_index ASC
                """,
                (task_ids,),
            )
            attempts_by_task: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in cursor.fetchall():
                attempts_by_task[int(row["task_id"])].append(dict(row))

            cursor.execute(
                """
                SELECT
                    LEAST(left_attempt_id, right_attempt_id) AS attempt_a,
                    GREATEST(left_attempt_id, right_attempt_id) AS attempt_b,
                    COUNT(*) FILTER (WHERE NOT is_seed) AS vote_count
                FROM gif_arena_votes
                GROUP BY 1, 2
                """
            )
            pair_vote_counts = {
                (int(row["attempt_a"]), int(row["attempt_b"])): int(row["vote_count"])
                for row in cursor.fetchall()
            }
            model_ratings = self._get_model_ratings(cursor)

            candidates: list[tuple[tuple[float, int, float, float], dict[str, object]]] = []
            for task_row in task_rows:
                task_id = int(task_row["task_id"])
                attempts = attempts_by_task[task_id]
                left_attempt, right_attempt, gap, pair_vote_count = select_relevant_attempt_pair(
                    attempts,
                    model_ratings,
                    pair_vote_counts,
                )
                priority = (
                    int(task_row["task_vote_count"]),
                    pair_vote_count,
                    gap,
                    random.random(),
                )
                candidates.append(
                    (
                        priority,
                        {
                            "task_id": task_id,
                            "source_interaction_id": int(task_row["source_interaction_id"]),
                            "task_text": str(task_row["task_text"]),
                            "task_vote_count": int(task_row["task_vote_count"]),
                            "pair_vote_count": pair_vote_count,
                            "participant_vote_count": participant_votes,
                            "remaining_tasks": len(task_rows) - 1,
                            "total_tasks": self.task_count,
                            "model_gap": round(gap, 2),
                            "left_attempt": left_attempt,
                            "right_attempt": right_attempt,
                        },
                    )
                )

            _, round_data = min(candidates, key=lambda item: item[0])
            if random.randint(0, 1) == 1:
                round_data["left_attempt"], round_data["right_attempt"] = (
                    round_data["right_attempt"],
                    round_data["left_attempt"],
                )

            round_data["left_gif_path"] = str(
                self.gif_dir / str(round_data["left_attempt"]["gif_id"])
            )
            round_data["right_gif_path"] = str(
                self.gif_dir / str(round_data["right_attempt"]["gif_id"])
            )
            return round_data

    def submit_vote(
        self,
        *,
        round_data: Mapping[str, object],
        participant_id: str,
        vote_type: str,
        requester_ip: str | None,
    ) -> dict[str, int]:
        participant_id = normalize_participant_id(participant_id)
        if vote_type not in ALLOWED_VOTE_TYPES:
            raise GifArenaError(f"Unexpected GIF arena vote type: {vote_type}")

        left_attempt = dict(round_data["left_attempt"])
        right_attempt = dict(round_data["right_attempt"])
        left_attempt_id = int(left_attempt["attempt_id"])
        right_attempt_id = int(right_attempt["attempt_id"])
        winner_attempt_id: int | None
        if vote_type == "leftvote":
            winner_attempt_id = left_attempt_id
        elif vote_type == "rightvote":
            winner_attempt_id = right_attempt_id
        else:
            winner_attempt_id = None

        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gif_arena_votes (
                    task_id,
                    left_attempt_id,
                    right_attempt_id,
                    winner_attempt_id,
                    participant_id,
                    vote_type,
                    requester_ip,
                    is_seed
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (task_id, participant_id, is_seed) DO NOTHING
                RETURNING id
                """,
                (
                    int(round_data["task_id"]),
                    left_attempt_id,
                    right_attempt_id,
                    winner_attempt_id,
                    participant_id,
                    vote_type,
                    requester_ip,
                ),
            )
            inserted_row = cursor.fetchone()
            if inserted_row is None:
                raise DuplicateVoteError(
                    "This participant has already voted on the current GIF task."
                )

            participant_vote_count = self._get_participant_vote_count(cursor, participant_id)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM gif_arena_tasks AS tasks
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM gif_arena_votes AS votes
                    WHERE votes.task_id = tasks.id
                        AND votes.participant_id = %s
                        AND NOT votes.is_seed
                )
                """,
                (participant_id,),
            )
            remaining_tasks = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT COUNT(*) FILTER (WHERE NOT is_seed)
                FROM gif_arena_votes
                WHERE task_id = %s
                """,
                (int(round_data["task_id"]),),
            )
            task_vote_count = int(cursor.fetchone()[0])

        return {
            "participant_vote_count": participant_vote_count,
            "remaining_tasks": remaining_tasks,
            "task_vote_count": task_vote_count,
        }

    @property
    def task_count(self) -> int:
        return len(self.unique_task_fingerprints)

    def _install_schema(self) -> None:
        with self.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gif_arena_tasks (
                    id BIGSERIAL PRIMARY KEY,
                    source_interaction_id INTEGER NOT NULL UNIQUE,
                    task_text TEXT NOT NULL,
                    source_vote_type TEXT NOT NULL,
                    source_recorded_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gif_arena_attempts (
                    id BIGSERIAL PRIMARY KEY,
                    task_id BIGINT NOT NULL REFERENCES gif_arena_tasks(id) ON DELETE CASCADE,
                    source_attempt_index INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    gif_id TEXT NOT NULL UNIQUE,
                    log_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (task_id, source_attempt_index)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gif_arena_votes (
                    id BIGSERIAL PRIMARY KEY,
                    task_id BIGINT NOT NULL REFERENCES gif_arena_tasks(id) ON DELETE CASCADE,
                    left_attempt_id BIGINT NOT NULL REFERENCES gif_arena_attempts(id) ON DELETE CASCADE,
                    right_attempt_id BIGINT NOT NULL REFERENCES gif_arena_attempts(id) ON DELETE CASCADE,
                    winner_attempt_id BIGINT REFERENCES gif_arena_attempts(id) ON DELETE CASCADE,
                    participant_id TEXT NOT NULL,
                    vote_type TEXT NOT NULL,
                    requester_ip TEXT,
                    is_seed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (task_id, participant_id, is_seed)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS gif_arena_votes_task_idx
                ON gif_arena_votes(task_id, participant_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS gif_arena_votes_pair_idx
                ON gif_arena_votes(left_attempt_id, right_attempt_id)
                """
            )

    def _bootstrap_seed_data(self) -> None:
        with self.connection() as connection, connection.cursor() as cursor:
            task_ids_by_fingerprint: dict[str, int] = {}
            for interaction in self.interactions:
                task_id = task_ids_by_fingerprint.get(interaction.task_fingerprint)
                if task_id is None:
                    cursor.execute(
                        """
                        INSERT INTO gif_arena_tasks (
                            source_interaction_id,
                            task_text,
                            source_vote_type,
                            source_recorded_at
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_interaction_id) DO UPDATE
                        SET
                            task_text = EXCLUDED.task_text,
                            source_vote_type = EXCLUDED.source_vote_type,
                            source_recorded_at = EXCLUDED.source_recorded_at
                        RETURNING id
                        """,
                        (
                            interaction.source_interaction_id,
                            interaction.task_text,
                            interaction.source_vote_type,
                            interaction.recorded_at,
                        ),
                    )
                    task_id = int(cursor.fetchone()[0])
                    task_ids_by_fingerprint[interaction.task_fingerprint] = task_id

                attempt_ids: dict[int, int] = {}
                for attempt in interaction.attempts:
                    cursor.execute(
                        """
                        INSERT INTO gif_arena_attempts (
                            task_id,
                            source_attempt_index,
                            model_name,
                            gif_id,
                            log_id,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (gif_id) DO UPDATE
                        SET
                            task_id = EXCLUDED.task_id,
                            source_attempt_index = EXCLUDED.source_attempt_index,
                            model_name = EXCLUDED.model_name,
                            log_id = EXCLUDED.log_id,
                            status = EXCLUDED.status
                        RETURNING id
                        """,
                        (
                            task_id,
                            attempt.source_attempt_index,
                            attempt.model_name,
                            attempt.gif_id,
                            attempt.log_id,
                            attempt.status,
                        ),
                    )
                    attempt_ids[attempt.source_attempt_index] = int(cursor.fetchone()[0])

                if len(attempt_ids) < 2:
                    continue

                left_attempt_id = attempt_ids[1]
                right_attempt_id = attempt_ids[2]
                if interaction.source_vote_type == "leftvote":
                    winner_attempt_id = left_attempt_id
                elif interaction.source_vote_type == "rightvote":
                    winner_attempt_id = right_attempt_id
                else:
                    winner_attempt_id = None

                cursor.execute(
                    """
                    INSERT INTO gif_arena_votes (
                        task_id,
                        left_attempt_id,
                        right_attempt_id,
                        winner_attempt_id,
                        participant_id,
                        vote_type,
                        requester_ip,
                        is_seed,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, TRUE, %s)
                    ON CONFLICT (task_id, participant_id, is_seed) DO NOTHING
                    """,
                    (
                        task_id,
                        left_attempt_id,
                        right_attempt_id,
                        winner_attempt_id,
                        f"{SEED_PARTICIPANT_ID}-{interaction.source_interaction_id}",
                        interaction.source_vote_type,
                        interaction.recorded_at or datetime.now(timezone.utc),
                    ),
                )

    def _get_model_ratings(self, cursor) -> dict[str, float]:
        cursor.execute(
            """
            SELECT
                left_attempts.model_name AS left_model_name,
                right_attempts.model_name AS right_model_name,
                votes.vote_type
            FROM gif_arena_votes AS votes
            JOIN gif_arena_attempts AS left_attempts
                ON left_attempts.id = votes.left_attempt_id
            JOIN gif_arena_attempts AS right_attempts
                ON right_attempts.id = votes.right_attempt_id
            ORDER BY votes.created_at ASC, votes.id ASC
            """
        )
        rows = cursor.fetchall()
        return compute_elo_ratings(
            (
                str(_row_value(row, "left_model_name", 0)),
                str(_row_value(row, "right_model_name", 1)),
                str(_row_value(row, "vote_type", 2)),
            )
            for row in rows
        )

    def _get_participant_vote_count(self, cursor, participant_id: str) -> int:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM gif_arena_votes
            WHERE participant_id = %s AND NOT is_seed
            """,
            (participant_id,),
        )
        return int(_scalar(cursor.fetchone()))

    def _validate_gif_files(self) -> None:
        if not self.gif_dir.exists():
            raise GifArenaError(f"GIF arena directory was not found: {self.gif_dir}")

        missing_files = [
            attempt.gif_id
            for interaction in self.interactions
            for attempt in interaction.attempts
            if not (self.gif_dir / attempt.gif_id).exists()
        ]
        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise GifArenaError(
                "The GIF arena is missing archived GIF assets. "
                f"Examples: {preview}"
            )


def _parse_recorded_at(raw_timestamp: str) -> datetime | None:
    try:
        return datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _build_task_fingerprint(
    task_text: str,
    attempts: Sequence[GifArenaAttempt],
) -> str:
    digest = hashlib.sha256()
    digest.update(task_text.strip().encode("utf-8"))
    for gif_id in sorted(attempt.gif_id for attempt in attempts):
        digest.update(b"\0")
        digest.update(gif_id.encode("utf-8"))
    return digest.hexdigest()


def _row_value(row, key: str, position: int):
    if isinstance(row, Mapping):
        return row[key]
    return row[position]


def _scalar(row) -> object:
    if isinstance(row, Mapping):
        return next(iter(row.values()))
    return row[0]
