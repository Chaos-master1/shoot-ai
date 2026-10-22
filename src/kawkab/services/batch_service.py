"""Batch processing service - overnight multi-match analysis queue.

Manages sequential analysis of multiple matches without user intervention.
Useful for processing weekend matches overnight or season-wide analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class BatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """A batch job for analyzing multiple matches."""

    id: int | None = None
    name: str = ""
    status: BatchStatus = BatchStatus.PENDING
    total_matches: int = 0
    completed_matches: int = 0
    failed_matches: int = 0
    match_ids: list[int] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str = ""


class BatchService:
    """Manages batch processing of multiple matches."""

    def __init__(self) -> None:
        self._running = False
        self._cancelled = False
        self._current_job: BatchJob | None = None
        logger.info("BatchService initialized")

    async def create_job(
        self,
        storage_service,
        name: str,
        match_ids: list[int],
        options: dict[str, Any] | None = None,
    ) -> BatchJob:
        """Create a new batch job in the database."""
        conn = storage_service._conn
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO batch_jobs (name, status, total_matches, match_ids, options)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                BatchStatus.PENDING.value,
                len(match_ids),
                json.dumps(match_ids),
                json.dumps(options or {}),
            ),
        )
        conn.commit()
        job_id = cursor.lastrowid or 0
        return BatchJob(
            id=job_id,
            name=name,
            status=BatchStatus.PENDING,
            total_matches=len(match_ids),
            match_ids=match_ids,
            options=options or {},
        )

    async def get_job(self, storage_service, job_id: int) -> BatchJob | None:
        """Get a batch job by ID."""
        conn = storage_service._conn
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_job(dict(row))

    async def list_jobs(self, storage_service, limit: int = 20) -> list[BatchJob]:
        """List recent batch jobs."""
        conn = storage_service._conn
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_job(dict(row)) for row in cursor.fetchall()]

    def _row_to_job(self, row: dict) -> BatchJob:
        return BatchJob(
            id=row["id"],
            name=row["name"],
            status=BatchStatus(row["status"]),
            total_matches=row["total_matches"],
            completed_matches=row["completed_matches"],
            failed_matches=row["failed_matches"],
            match_ids=json.loads(row["match_ids"] or "[]"),
            options=json.loads(row["options"] or "{}"),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            error_message=row.get("error_message", ""),
        )

    async def run_job(
        self,
        storage_service,
        bridge,
        job_id: int,
        progress_callback=None,
    ) -> BatchJob:
        """Execute a batch job sequentially.

        Args:
            storage_service: For DB updates
            bridge: Bridge instance with analyze_match method
            job_id: Batch job ID to run
            progress_callback: Called with (completed, total, match_id, status)

        Returns:
            Updated BatchJob
        """
        job = await self.get_job(storage_service, job_id)
        if not job:
            raise ValueError(f"Batch job {job_id} not found")
        if job.status == BatchStatus.RUNNING:
            raise ValueError(f"Batch job {job_id} is already running")

        self._running = True
        self._current_job = job

        conn = storage_service._conn
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batch_jobs SET status = ?, started_at = datetime('now') WHERE id = ?",
            (BatchStatus.RUNNING.value, job_id),
        )
        conn.commit()
        job.status = BatchStatus.RUNNING

        logger.info(f"Starting batch job {job_id}: {job.name} ({job.total_matches} matches)")

        if self._cancelled:
            logger.info(f"Batch job {job_id} cancelled before start")
            cursor.execute(
                "UPDATE batch_jobs SET status = ? WHERE id = ?",
                (BatchStatus.CANCELLED.value, job_id),
            )
            conn.commit()
            job.status = BatchStatus.CANCELLED
            self._running = False
            return job

        for match_id in job.match_ids:
            if self._cancelled:
                logger.info(f"Batch job {job_id} cancelled")
                cursor.execute(
                    "UPDATE batch_jobs SET status = ? WHERE id = ?",
                    (BatchStatus.CANCELLED.value, job_id),
                )
                conn.commit()
                job.status = BatchStatus.CANCELLED
                self._running = False
                return job

            try:
                match = await storage_service.get_match(match_id)
                if not match or not match.get("video_path"):
                    raise ValueError(f"Match {match_id} has no video path")

                logger.info(f"Batch job {job_id}: analyzing match {match_id}")
                result_json = await bridge.analyze_match(match_id, match["video_path"])
                result = json.loads(result_json)

                if result.get("error"):
                    raise RuntimeError(result["error"])

                job.completed_matches += 1
                cursor.execute(
                    "UPDATE batch_jobs SET completed_matches = ? WHERE id = ?",
                    (job.completed_matches, job_id),
                )
                conn.commit()

                if progress_callback:
                    await progress_callback(
                        job.completed_matches,
                        job.total_matches,
                        match_id,
                        "ok",
                    )

            except Exception as e:
                logger.error(f"Batch job {job_id}: match {match_id} failed: {e}")
                job.failed_matches += 1
                cursor.execute(
                    "UPDATE batch_jobs SET failed_matches = ?, error_message = ? WHERE id = ?",
                    (job.failed_matches, str(e)[:500], job_id),
                )
                conn.commit()

                if progress_callback:
                    await progress_callback(
                        job.completed_matches,
                        job.total_matches,
                        match_id,
                        "error",
                    )

        final_status = BatchStatus.COMPLETED if job.failed_matches == 0 else BatchStatus.FAILED
        cursor.execute(
            "UPDATE batch_jobs SET status = ?, completed_at = datetime('now') WHERE id = ?",
            (final_status.value, job_id),
        )
        conn.commit()
        job.status = final_status

        self._running = False
        self._cancelled = False
        self._current_job = None
        logger.info(
            f"Batch job {job_id} finished: {job.completed_matches}/{job.total_matches} matches"
        )
        return job

    def cancel(self) -> None:
        """Cancel the currently running batch job."""
        self._cancelled = True
        logger.info("Batch cancellation requested")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_job(self) -> BatchJob | None:
        return self._current_job
