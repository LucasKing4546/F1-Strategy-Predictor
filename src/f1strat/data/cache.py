from pathlib import Path

import fastf1

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def enable_cache(cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache_dir))
    return cache_dir
