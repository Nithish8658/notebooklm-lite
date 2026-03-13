import re
import random
from typing import Dict, Optional
from pydantic import BaseModel


# ============================================
# Data Model
# ============================================

class EnrichedSegment(BaseModel):
    speaker: str
    original_text: str
    emotion: str
    config: Dict[str, str]


# ============================================
# Prosody Planner
# ============================================

class ProsodyPlanner:
    """
    Episode-aware conversational prosody engine.
    Models tension, inertia, dominance, and escalation.
    Fully compatible with existing pipeline.
    """

    EMOTION_CONFIG = {
        "EXCITED":     {"rate": 15,  "pitch": 6,   "volume": 10},
        "INQUISITIVE": {"rate": 6,   "pitch": 10,  "volume": 0},
        "REFLECTIVE":  {"rate": -8,  "pitch": -5,  "volume": -5},
        "SERIOUS":     {"rate": -5,  "pitch": -3,  "volume": 6},
        "NEUTRAL":     {"rate": 0,   "pitch": 0,   "volume": 0},
    }

    INTENT_PATTERNS = {
        "EXCITED": [r"\!+", r"\bwow\b", r"\bamazing\b"],
        "INQUISITIVE": [r"\?+", r"\breally\b", r"\bhow\b"],
        "REFLECTIVE": [r"\.\.\.", r"\bhmm\b"],
        "SERIOUS": [r"\bimportant\b", r"\bcritical\b"]
    }

    def __init__(self):
        self.prev_emotion: Optional[str] = None
        self.prev_speaker: Optional[str] = None
        self.emotion_momentum: float = 0.0
        self.tension_score: float = 0.0
        self.dominance: Dict[str, int] = {"Host 1": 0, "Host 2": 0}

    # ============================================
    # Emotion Detection
    # ============================================

    def detect_emotion(self, text: str) -> str:
        lower = text.lower()
        scores = {k: 0 for k in self.INTENT_PATTERNS}

        for emotion, patterns in self.INTENT_PATTERNS.items():
            for p in patterns:
                scores[emotion] += len(re.findall(p, lower))

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "NEUTRAL"

    # ============================================
    # Emotional Inertia
    # ============================================

    def apply_emotional_inertia(self, emotion: str, config: Dict[str, int]) -> Dict[str, int]:
        adjusted = config.copy()

        if self.prev_emotion and self.prev_emotion != emotion:
            # Resist abrupt flips
            adjusted["rate"] = int(adjusted["rate"] * 0.8)
            adjusted["pitch"] = int(adjusted["pitch"] * 0.8)

        return adjusted

    # ============================================
    # Momentum Modeling
    # ============================================

    def update_momentum(self, emotion: str):
        if emotion == "EXCITED":
            self.emotion_momentum += 0.2
        elif emotion == "SERIOUS":
            self.tension_score += 0.3
        else:
            self.emotion_momentum *= 0.9
            self.tension_score *= 0.9

    # ============================================
    # Speaker Dominance
    # ============================================

    def apply_dominance_balance(self, speaker: str, config: Dict[str, int]) -> Dict[str, int]:
        adjusted = config.copy()

        self.dominance[speaker] += 1

        if abs(self.dominance["Host 1"] - self.dominance["Host 2"]) > 3:
            # reduce energy for dominant speaker
            adjusted["rate"] -= 3

        return adjusted

    # ============================================
    # Emphasis Multiplier
    # ============================================

    def detect_emphasis_multiplier(self, text: str) -> float:
        multiplier = 1.0

        if re.search(r"\b[A-Z]{3,}\b", text):
            multiplier += 0.2

        if text.count("!") > 1:
            multiplier += 0.2

        return min(multiplier, 1.5)

    # ============================================
    # Micro Variation
    # ============================================

    def apply_micro_variation(self, config: Dict[str, int]) -> Dict[str, int]:
        adjusted = config.copy()

        adjusted["rate"] += random.randint(-2, 2)
        adjusted["pitch"] += random.randint(-1, 1)
        adjusted["volume"] += random.randint(-1, 1)

        return adjusted

    # ============================================
    # Normalize
    # ============================================

    def normalize(self, config: Dict[str, int]) -> Dict[str, str]:
        return {
            "rate": f"{max(min(config['rate'], 25), -25):+d}%",
            "pitch": f"{max(min(config['pitch'], 20), -20):+d}Hz",
            "volume": f"{max(min(config['volume'], 20), -20):+d}%"
        }

    # ============================================
    # Main Entry
    # ============================================

    def plan_segment(self, speaker: str, text: str) -> EnrichedSegment:

        emotion = self.detect_emotion(text)
        base = self.EMOTION_CONFIG.get(emotion, self.EMOTION_CONFIG["NEUTRAL"]).copy()

        # Emphasis
        multiplier = self.detect_emphasis_multiplier(text)
        for k in base:
            base[k] = int(base[k] * multiplier)

        # Emotional inertia
        base = self.apply_emotional_inertia(emotion, base)

        # Momentum update
        self.update_momentum(emotion)

        # Speaker balance
        base = self.apply_dominance_balance(speaker, base)

        # Micro human variation
        base = self.apply_micro_variation(base)

        final_config = self.normalize(base)

        # Update previous state
        self.prev_emotion = emotion
        self.prev_speaker = speaker

        return EnrichedSegment(
            speaker=speaker,
            original_text=text,
            emotion=emotion,
            config=final_config
        )


# Singleton fallback
planner = ProsodyPlanner()
