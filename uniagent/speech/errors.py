class SpeechError(RuntimeError):
    """Base error for optional speech support."""


class SpeechConfigurationError(SpeechError):
    """Raised when speech config cannot drive the requested backend."""
