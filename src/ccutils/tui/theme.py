"""Color constants, prompt_toolkit Style, and semantic style mappings.

Single source of truth for all visual styling in the TUI.
"""

from prompt_toolkit.styles import Style

# -- Semantic role -> prompt_toolkit style string (for questionary FormattedText) --
STYLES = {
    "temporal": "fg:#e5c07b",  # warm yellow -- dates, times, durations
    "identity": "fg:#61afef",  # blue -- project names, branches
    "identity.bold": "fg:#61afef bold",
    "model": "fg:#c678dd",  # magenta -- model names
    "model.opus": "fg:#c678dd bold",
    "model.sonnet": "fg:#c678dd",
    "model.haiku": "fg:#c678dd italic",
    "metric": "fg:#98c379",  # green -- counts, sizes, numeric measures
    "primary": "",  # default -- summaries, main content
    "secondary": "fg:#5c6370",  # gray -- row numbers, old sessions
    "chain": "fg:#e5c07b italic",  # italic yellow -- chain indicators
}

# -- Semantic role -> Rich markup style (for Rich tables) --
RICH_STYLES = {
    "temporal": "yellow",
    "identity": "cyan",
    "identity.bold": "cyan bold",
    "model": "magenta",
    "model.opus": "magenta bold",
    "model.sonnet": "magenta",
    "model.haiku": "magenta italic",
    "metric": "green",
    "primary": "white",
    "secondary": "dim",
    "chain": "italic yellow",
}

# Model family -> style key mapping
MODEL_FAMILIES = {
    "opus": "model.opus",
    "sonnet": "model.sonnet",
    "haiku": "model.haiku",
}


def model_style_key(model_short: str) -> str:
    """Return the STYLES/RICH_STYLES key for a model short name.

    Args:
        model_short: Shortened model name like "opus-4.6", "sonnet-4.5"

    Returns:
        Style key like "model.opus", or "model" as fallback.
    """
    lower = model_short.lower()
    for family, key in MODEL_FAMILIES.items():
        if lower.startswith(family):
            return key
    return "model"


def questionary_style() -> Style:
    """Build a prompt_toolkit Style for questionary checkbox/select chrome.

    Returns a Style that colors the pointer, highlight, selected markers,
    answer text, and instruction text.
    """
    return Style(
        [
            ("qmark", "fg:#61afef bold"),  # question mark
            ("question", "bold"),  # question text
            ("pointer", "fg:#61afef bold"),  # >> pointer
            ("highlighted", "fg:#61afef bold"),  # current item highlight
            ("selected", "fg:#98c379"),  # selected checkbox marker
            ("answer", "fg:#98c379 bold"),  # answered value
            ("instruction", "fg:#5c6370"),  # instruction text
        ]
    )
