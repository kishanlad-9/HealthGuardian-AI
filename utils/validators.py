"""Shared validation helpers - kept here (not in authentication/) since
future features (e.g. profile edit) will need the same rules."""
import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    """Basic email shape check - not a full RFC 5322 validator, which is
    intentionally out of scope for a portfolio project's signup form."""
    return bool(EMAIL_PATTERN.match(email.strip()))


def is_valid_username(username: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message). Empty error_message if valid."""
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must be 30 characters or fewer."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, ""


def is_valid_password(password: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message). Minimum bar for a portfolio app -
    length + character variety, not a full entropy/breach check."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must include at least one letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one number."
    return True, ""
