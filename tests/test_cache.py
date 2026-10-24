import fastf1

from f1strat.data.cache import CACHE_DIR, enable_cache


def test_enable_cache_creates_dir_and_points_fastf1_at_it(tmp_path):
    target = tmp_path / "nested" / "cache"

    result = enable_cache(target)

    assert result == target
    assert target.is_dir()
    assert fastf1.Cache.get_cache_info()[0] == str(target)


def test_default_cache_dir_is_repo_data_cache():
    assert CACHE_DIR.parts[-2:] == ("data", "cache")
    assert (CACHE_DIR.parents[1] / "pyproject.toml").is_file()
