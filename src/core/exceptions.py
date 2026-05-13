"""Custom exception hierarchy for Finnie."""


class FinnieError(Exception):
    """Base exception for all Finnie errors."""


class ConfigError(FinnieError):
    """Raised when configuration is invalid or missing."""


class RateLimitError(FinnieError):
    """Raised when an API rate limit is hit."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(FinnieError):
    """Raised when an external API call fails."""

    def __init__(self, message: str, service: str = "", status_code: int = 0):
        super().__init__(message)
        self.service = service
        self.status_code = status_code


class QuotaExceededError(APIError):
    """Raised when a daily/monthly API quota is exhausted."""


class RAGError(FinnieError):
    """Raised when RAG retrieval or indexing fails."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class AgentError(FinnieError):
    """Raised when an agent fails to produce a response."""

    def __init__(self, message: str, agent_name: str = ""):
        super().__init__(message)
        self.agent_name = agent_name


class WorkflowError(FinnieError):
    """Raised when the LangGraph workflow encounters an unrecoverable error."""
