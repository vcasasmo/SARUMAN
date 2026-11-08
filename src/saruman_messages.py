"""Small, optional terminal messages; science/logging stays precise."""

MESSAGES = {
    "config_ok": "The configuration is sound. The seeing stone is ready.",
    "paths_cached": "These paths are already known to me.",
    "paths_new": "Then I shall forge the paths anew.",
    "prepare_done": "The drum has yielded its secrets. They are sealed within the archive.",
    "reconstruction_done": "The posterior has spoken.",
    "experimental": "You venture beyond the realm of proven knowledge.",
}


def announce(event: str, *, enabled: bool = True) -> None:
    """Print flavor text only for recognized events."""
    if enabled:
        print(f'SARUMAN: "{MESSAGES[event]}"')
