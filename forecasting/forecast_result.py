from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ForecastStep:

    step: int

    state: List[float]

    attack_probability: float

    risk_score: float

    risk_level: str

    mitre_stage: str

    mitre_confidence: float


@dataclass
class ForecastResult:

    current_attack_probability: float

    steps: List[ForecastStep]

    summary: Dict[str, Any]