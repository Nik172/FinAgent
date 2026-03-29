import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents.agent_alpha import run_agent_alpha
from agents.agent_beta import run_agent_beta
from agents.agent_gamma import run_agent_gamma, load_finbert_onnx_pipeline
from master_node import run_master_node

# ── startup: load heavy models once ───────────────────────────────────────────
# FinBERT ONNX takes ~5 seconds to load. This runs when the orchestrator module
# is first imported — not on every request.

FINBERT = load_finbert_onnx_pipeline()

# Thread pool for running blocking agent functions concurrently.
# Three workers — one per agent.
EXECUTOR = ThreadPoolExecutor(max_workers=3)


# ── parallel agent dispatch ────────────────────────────────────────────────────

async def run_agents_in_parallel(ticker: str) -> tuple[dict, dict, dict]:
    """
    Fires all three agents concurrently using asyncio + ThreadPoolExecutor.
    Each agent is a blocking function (PDF I/O, HTTP calls, model inference),
    so we offload them to threads to prevent blocking the event loop.
    Returns a tuple of (alpha_result, beta_result, gamma_result).
    """
    event_loop = asyncio.get_event_loop()

    alpha_future = event_loop.run_in_executor(
        EXECUTOR,
        run_agent_alpha,
        ticker,
    )
    beta_future = event_loop.run_in_executor(
        EXECUTOR,
        run_agent_beta,
        ticker,
    )
    gamma_future = event_loop.run_in_executor(
        EXECUTOR,
        lambda: run_agent_gamma(ticker, FINBERT),
    )

    alpha_result, beta_result, gamma_result = await asyncio.gather(
        alpha_future,
        beta_future,
        gamma_future,
    )
    return alpha_result, beta_result, gamma_result


# ── response assembly ──────────────────────────────────────────────────────────

def assemble_final_response(
    ticker: str,
    alpha_result: dict,
    beta_result: dict,
    gamma_result: dict,
    master_result: dict,
) -> dict:
    """
    Bundles the Master Node verdict and all agent explanations into
    a single response dictionary for the UI to consume.
    """
    any_fallback = any(
        agent["status"] == "fallback"
        for agent in [alpha_result, beta_result, gamma_result]
    )

    return {
        "ticker": ticker,
        "verdict": master_result["verdict"],
        "confidence": master_result["confidence"],
        "composite_score": master_result["composite_score"],
        "data_quality": "partial" if any_fallback else "full",
        "agents": {
            "fundamental": {
                "score":      alpha_result["normalized_score"],
                "status":     alpha_result["status"],
                "key_driver": alpha_result["raw_metrics"].get("key_driver", ""),
            },
            "technical": {
                "score":      beta_result["normalized_score"],
                "status":     beta_result["status"],
                "key_driver": beta_result["raw_metrics"].get("key_driver", ""),
            },
            "sentiment": {
                "score":      gamma_result["normalized_score"],
                "status":     gamma_result["status"],
                "key_driver": gamma_result["raw_metrics"].get("key_driver", ""),
            },
        },
        "weights": master_result["weights_used"],
    }


# ── main entry point ───────────────────────────────────────────────────────────

async def analyze(ticker: str) -> dict:
    """
    Top-level entry point called by the FastAPI route handler.
    Accepts a ticker string like "RELIANCE.NS".
    Always returns a complete response dict — never raises.
    """
    try:
        alpha_result, beta_result, gamma_result = await run_agents_in_parallel(ticker)

        master_result = run_master_node(
            fundamental_score=alpha_result["normalized_score"],
            technical_score=beta_result["normalized_score"],
            sentiment_score=gamma_result["normalized_score"],
        )

        return assemble_final_response(
            ticker,
            alpha_result,
            beta_result,
            gamma_result,
            master_result,
        )

    except Exception as error:
        return {
            "ticker": ticker,
            "verdict": "ERROR",
            "confidence": 0.0,
            "composite_score": 0.0,
            "data_quality": "failed",
            "error": str(error),
            "agents": {},
            "weights": {},
        }