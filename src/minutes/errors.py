class MinutesError(Exception):
    pass


class ConfigError(MinutesError):
    pass


class DatabaseError(MinutesError):
    pass


class MigrationError(MinutesError):
    pass


class ValidationError(MinutesError):
    pass


class NotFoundError(MinutesError):
    pass


class SourceError(MinutesError):
    pass


class StageError(MinutesError):
    pass


class PermanentStageError(MinutesError):
    pass


class CorpusError(MinutesError):
    pass


class LLMTransportError(MinutesError):
    pass


class LLMQuotaError(MinutesError):
    pass


class LLMOutputError(MinutesError):
    pass


class LLMReplayMiss(MinutesError):
    pass


class BudgetExceeded(MinutesError):
    pass


class ToolError(MinutesError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LabelValidationError(MinutesError):
    pass


class LabelRuleError(MinutesError):
    pass


class LabelsFrozenError(MinutesError):
    pass


class LabelsNotFrozenError(MinutesError):
    pass


class GateStaleError(MinutesError):
    pass
