"""HSV parameter persistence via JSON.

Default values match the reference C driver's initial color thresholds.
Parameters are saved to and loaded from calibration_data/hsv_params.json.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("display.config")

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "calibration_data" / "hsv_params.json"

DEFAULT_HSV: dict[str, int] = {
    "H_min": 0,
    "S_min": 80,
    "V_min": 80,
    "H_max": 180,
    "S_max": 255,
    "V_max": 255,
}


def load_hsv(path: Path | str = DEFAULT_PATH) -> dict[str, int]:
    """Load HSV params from disk, returning defaults on any error."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
        # Merge with defaults so missing keys are filled in
        result = dict(DEFAULT_HSV)
        result.update({k: int(v) for k, v in data.items() if k in DEFAULT_HSV})
        logger.info("Loaded HSV params from %s", path)
        return result
    except FileNotFoundError:
        logger.info("No HSV config at %s, using defaults", path)
        return dict(DEFAULT_HSV)
    except Exception:
        logger.exception("Failed to load HSV params from %s", path)
        return dict(DEFAULT_HSV)


def save_hsv(params: dict[str, int], path: Path | str = DEFAULT_PATH) -> None:
    """Persist HSV params to disk."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(params, indent=2))
        logger.info("Saved HSV params to %s", path)
    except Exception:
        logger.exception("Failed to save HSV params to %s", path)
