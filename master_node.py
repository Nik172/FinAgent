# ── constants ──────────────────────────────────────────────────────────────────

# Weights must sum to 1.0
# Fundamental carries the most weight — balance sheet health is the foundation.
# Technical is the weakest signal for long-term decisions.
WEIGHT_FUNDAMENTAL = 0.45
WEIGHT_TECHNICAL   = 0.25
WEIGHT_SENTIMENT   = 0.30

# Verdict thresholds (composite score is on a 0–10 scale)
THRESHOLD_BUY  = 6.5   # composite >= 6.5 → BUY
THRESHOLD_SELL = 4.0   # composite <= 4.0 → SELL
                        # everything in between → HOLD


# ── core logic ─────────────────────────────────────────────────────────────────

def compute_composite_score(
    fundamental_score: float,
    technical_score: float,
    sentiment_score: float,
) -> float:
    composite = (
        fundamental_score * WEIGHT_FUNDAMENTAL +
        technical_score   * WEIGHT_TECHNICAL   +
        sentiment_score   * WEIGHT_SENTIMENT
    )
    return round(composite, 2)


def compute_confidence(composite_score: float, verdict: str) -> float:
    """
    Expresses how far the composite score is from the nearest threshold,
    mapped to a 0–100% confidence value.

    BUY  confidence: how far above 6.5 we are, scaled to the remaining 3.5 points.
    SELL confidence: how far below 4.0 we are, scaled to the available 4.0 points.
    HOLD confidence: how close to the midpoint (5.25) we are.
    """
    if verdict == "BUY":
        distance = composite_score - THRESHOLD_BUY          # 0 to 3.5
        confidence = (distance / (10.0 - THRESHOLD_BUY)) * 100
    elif verdict == "SELL":
        distance = THRESHOLD_SELL - composite_score         # 0 to 4.0
        confidence = (distance / THRESHOLD_SELL) * 100
    else:  # HOLD
        midpoint = (THRESHOLD_BUY + THRESHOLD_SELL) / 2.0  # 5.25
        distance_from_mid = abs(composite_score - midpoint)
        max_hold_distance = (THRESHOLD_BUY - THRESHOLD_SELL) / 2.0  # 1.25
        confidence = (1.0 - distance_from_mid / max_hold_distance) * 100

    return round(min(100.0, max(0.0, confidence)), 1)


def determine_verdict(composite_score: float) -> str:
    if composite_score >= THRESHOLD_BUY:
        return "BUY"
    elif composite_score <= THRESHOLD_SELL:
        return "SELL"
    else:
        return "HOLD"


# ── main entry point ───────────────────────────────────────────────────────────

def run_master_node(
    fundamental_score: float,
    technical_score: float,
    sentiment_score: float,
) -> dict:
    """
    Entry point called by the orchestrator.
    Takes the three normalized agent scores (each 0–10) and returns a verdict.
    """
    composite_score = compute_composite_score(
        fundamental_score,
        technical_score,
        sentiment_score,
    )
    verdict = determine_verdict(composite_score)
    confidence = compute_confidence(composite_score, verdict)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "composite_score": composite_score,
        "weights_used": {
            "fundamental": WEIGHT_FUNDAMENTAL,
            "technical":   WEIGHT_TECHNICAL,
            "sentiment":   WEIGHT_SENTIMENT,
        },
    }