"""Settings module for the project."""

from pathlib import Path

from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "settings.toml",
        ".secrets.toml",
    ],
    root_path=Path(__file__).parents[1],
    silent=False,
    merge_enabled=True,
)
