from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StructuredOutputsParams:
    json: Any = None


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 1.0
    n: int = 1
    structured_outputs: Optional[StructuredOutputsParams] = None
