"""Exception hierarchy shared across FocusFlow ML.

Every expected, user-facing failure derives from :class:`FocusFlowError` so the
CLI can report it cleanly. Anything else is a genuine bug and is allowed to
propagate.
"""

from __future__ import annotations


class FocusFlowError(Exception):
    """Base class for expected application errors."""


class ConfigError(FocusFlowError):
    """Configuration could not be loaded."""


class ValidationError(FocusFlowError):
    """User or data input failed validation."""


class NotFoundError(FocusFlowError):
    """A requested record does not exist."""


class DuplicateError(FocusFlowError):
    """A record would violate a uniqueness rule."""


class DatabaseError(FocusFlowError):
    """The database is missing, uninitialised or failed an operation."""


class DataError(FocusFlowError):
    """A dataset is missing, malformed or unusable for training."""


class ModelError(FocusFlowError):
    """A persisted model is missing, corrupted or incompatible."""
