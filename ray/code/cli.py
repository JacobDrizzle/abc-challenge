"""Small helpers for asking the user questions and printing sizes."""


def ask_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    return int(raw) if raw.lstrip("-").isdigit() else default


def ask_float(prompt: str) -> float | None:
    raw = input(prompt).strip()
    try:
        return float(raw)
    except ValueError:
        return None


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
