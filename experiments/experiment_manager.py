import json
import os
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class RunTracker:
    """
    Experimental Run Tracker for systematic research logging.
    Tracks metadata, metrics, and artifact paths for every research run.
    """
    def __init__(self, registry_path: str = "experiments/run_registry.json"):
        self.registry_path = Path(registry_path)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, "w") as f:
                json.dump([], f)

    def log_run(self, 
                experiment_id: str, 
                config: Dict[str, Any], 
                metrics: Dict[str, Any], 
                artifacts: Optional[Dict[str, str]] = None):
        """Logs a single research run to the registry."""
        run_data = {
            "run_id": f"{experiment_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts or {}
        }

        with open(self.registry_path, "r") as f:
            runs = json.load(f)

        runs.append(run_data)

        with open(self.registry_path, "w") as f:
            json.dump(runs, f, indent=2)
            
        return run_data["run_id"]

    def get_latest_run(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent run for a specific experiment ID."""
        with open(self.registry_path, "r") as f:
            runs = json.load(f)
            
        matching_runs = [r for r in runs if r["run_id"].startswith(experiment_id)]
        return matching_runs[-1] if matching_runs else None
