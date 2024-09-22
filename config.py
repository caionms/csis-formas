from pathlib import Path

from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "settings.toml",
        ".secrets.toml",
    ],
    root_path=Path(__file__).parent,
    silent=False,
    merge_enabled=True,
)
