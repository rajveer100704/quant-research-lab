"""
Job Runner
==========
Simple task scheduler for periodic data collection, feature updates,
and model retraining using APScheduler.

Jobs:
- Hourly market data sync
- Daily feature store update
- Weekly model retraining
"""

from __future__ import annotations

from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class JobRunner:
    """
    Lightweight scheduler for recurring platform tasks.

    Uses APScheduler if available, otherwise provides a simple
    loop-based fallback.

    Example
    -------
    >>> runner = JobRunner()
    >>> runner.add_job(collect_data, trigger="interval", hours=1, job_id="data_sync")
    >>> runner.add_job(update_features, trigger="cron", hour=0, job_id="feature_update")
    >>> runner.start()
    """

    def __init__(self) -> None:
        self._scheduler = None
        self._jobs: dict[str, dict] = {}

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self._scheduler = BackgroundScheduler()
            logger.info("APScheduler initialised")
        except ImportError:
            logger.warning("APScheduler not installed — scheduler in manual mode")

    def add_job(
        self,
        func: Callable,
        trigger: str = "interval",
        job_id: Optional[str] = None,
        **trigger_args,
    ) -> None:
        """
        Register a scheduled job.

        Parameters
        ----------
        func : Callable
            Function to execute.
        trigger : str
            ``"interval"`` or ``"cron"``.
        job_id : str, optional
            Unique identifier for the job.
        **trigger_args
            Passed to APScheduler (e.g. ``hours=1``, ``hour=0``).
        """
        job_id = job_id or func.__name__

        if self._scheduler is not None:
            self._scheduler.add_job(func, trigger=trigger, id=job_id, **trigger_args)
            logger.info("Job registered: %s (%s)", job_id, trigger)
        else:
            self._jobs[job_id] = {"func": func, "trigger": trigger, "args": trigger_args}
            logger.info("Job stored (manual mode): %s", job_id)

    def start(self) -> None:
        """Start the scheduler."""
        if self._scheduler is not None:
            self._scheduler.start()
            logger.info("Scheduler started with %d jobs", len(self._scheduler.get_jobs()))
        else:
            logger.warning("No scheduler available. Run jobs manually.")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")

    def run_all_once(self) -> None:
        """Execute all registered jobs once (manual mode)."""
        for job_id, job in self._jobs.items():
            logger.info("Running job: %s", job_id)
            try:
                job["func"]()
            except Exception as exc:
                logger.error("Job %s failed: %s", job_id, exc)

    def list_jobs(self) -> list[str]:
        """List registered job IDs."""
        if self._scheduler is not None:
            return [j.id for j in self._scheduler.get_jobs()]
        return list(self._jobs.keys())
