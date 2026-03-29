import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator import analyze
from agents.agent_gamma import TICKER_TO_COMPANY_NAME

# ── app setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Analysis API",
    description="Multi-agent stock analysis backend — Fundamental, Technical, Sentiment.",
    version="1.0.0",
)

# Allow the Streamlit frontend to call this backend without CORS errors.
# In production, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Lightweight liveness check.
    Frontend calls this on startup to confirm the backend is reachable.
    """
    return {"status": "ok"}


@app.get("/tickers")
def get_supported_tickers():
    """
    Returns the list of tickers the system supports.
    Frontend uses this to populate the dropdown — prevents typo failures.
    """
    tickers = [
        {"ticker": ticker, "company": company}
        for ticker, company in TICKER_TO_COMPANY_NAME.items()
    ]
    return {"supported_tickers": tickers}


@app.post("/analyze")
async def analyze_ticker(request: AnalyzeRequest):
    """
    Core route. Accepts a ticker, runs all three agents in parallel,
    passes scores to the Master Node, and returns the full verdict.

    Example request body:
        { "ticker": "RELIANCE.NS" }

    Example response:
        {
            "ticker": "RELIANCE.NS",
            "verdict": "BUY",
            "confidence": 74.3,
            "composite_score": 7.82,
            "data_quality": "full",
            "agents": {
                "fundamental": { "score": 8.5, "status": "success", "key_driver": "..." },
                "technical":   { "score": 5.1, "status": "success", "key_driver": "..." },
                "sentiment":   { "score": 9.1, "status": "success", "key_driver": "..." }
            },
            "weights": { "fundamental": 0.45, "technical": 0.25, "sentiment": 0.30 }
        }
    """
    ticker = request.ticker.strip().upper()

    if ticker not in TICKER_TO_COMPANY_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' is not supported. Call GET /tickers for the full list.",
        )

    result = await analyze(ticker)

    if result.get("verdict") == "ERROR":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {result.get('error', 'Unknown error')}",
        )

    return result