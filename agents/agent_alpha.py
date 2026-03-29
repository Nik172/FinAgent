import os
import json
import pdfplumber
from groq import Groq

# ── constants ──────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PDF_FOLDER = "data"
PDF_PAGES_TO_SCAN = 3

# Normalization benchmarks — adjust these per sector if needed
DE_RATIO_WORST = 3.0        # D/E at or above this → score 0
MARGIN_BEST = 30.0          # profit margin at or above this → score 10
FCF_BEST_CRORES = 5000.0    # FCF at or above this → score 10

# Custom weights (must sum to 1.0)
WEIGHT_DEBT_TO_EQUITY = 0.45
WEIGHT_PROFIT_MARGIN = 0.35
WEIGHT_FREE_CASH_FLOW = 0.20

SYSTEM_PROMPT = """You are a quantitative financial analyst.
You will be given raw text extracted from a quarterly earnings PDF.
Extract exactly three financial metrics and write one key driver sentence.
Respond with ONLY a valid JSON object — no preamble, no explanation, no markdown fences.
The JSON must have exactly these keys:
  - debt_to_equity: float (e.g. 0.45). Lower is safer. Typical range 0.0 to 3.0.
  - profit_margin: float as a percentage value (e.g. 18.5 means 18.5%). Can be negative.
  - free_cash_flow_crores: float in crores INR (e.g. 3200.0). Can be negative.
  - key_driver: a single sentence describing the primary financial strength or weakness.
If a metric cannot be found in the text, use null for that field."""


# ── stage 1: pdf extraction ────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    extracted_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_read = pdf.pages[:PDF_PAGES_TO_SCAN]
        for page in pages_to_read:
            page_text = page.extract_text()
            if page_text:
                extracted_chunks.append(page_text)

            page_tables = page.extract_tables()
            for table in page_tables:
                for row in table:
                    cleaned_cells = [cell if cell is not None else "" for cell in row]
                    extracted_chunks.append(" | ".join(cleaned_cells))

    if not extracted_chunks:
        raise ValueError(f"No text could be extracted from {pdf_path}")

    return "\n".join(extracted_chunks)


# ── stage 2: llm metric extraction ────────────────────────────────────────────

def query_llm_for_metrics(raw_text: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Earnings report text:\n\n{raw_text}"},
        ],
    )

    raw_output = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps output despite instructions
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        raw_output = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    parsed = json.loads(raw_output)

    # Replace any null values with neutral defaults so normalization never breaks
    if parsed.get("debt_to_equity") is None:
        parsed["debt_to_equity"] = 1.0
    if parsed.get("profit_margin") is None:
        parsed["profit_margin"] = 0.0
    if parsed.get("free_cash_flow_crores") is None:
        parsed["free_cash_flow_crores"] = 0.0
    if parsed.get("key_driver") is None:
        parsed["key_driver"] = "Insufficient data to determine key driver."

    return parsed


# ── stage 3: normalization ─────────────────────────────────────────────────────

def normalize_debt_to_equity(de_ratio: float) -> float:
    """
    D/E 0.0  → score 10  (no debt, very safe)
    D/E 1.5  → score 5   (moderate leverage)
    D/E 3.0+ → score 0   (heavily leveraged)
    Linear interpolation between 0 and DE_RATIO_WORST.
    """
    clamped = max(0.0, min(de_ratio, DE_RATIO_WORST))
    score = 10.0 * (1.0 - clamped / DE_RATIO_WORST)
    return round(score, 2)


def normalize_profit_margin(margin_percent: float) -> float:
    """
    Margin 0%   → score 0
    Margin 30%+ → score 10
    Negative margins are clamped to 0.
    """
    clamped = max(0.0, min(margin_percent, MARGIN_BEST))
    score = (clamped / MARGIN_BEST) * 10.0
    return round(score, 2)


def normalize_free_cash_flow(fcf_crores: float) -> float:
    """
    FCF 5000+ crores → score 10
    FCF 0            → score 0
    Negative FCF is clamped to 0 (not penalized beyond neutral).
    """
    clamped = max(0.0, min(fcf_crores, FCF_BEST_CRORES))
    score = (clamped / FCF_BEST_CRORES) * 10.0
    return round(score, 2)


def compute_weighted_score(metrics: dict) -> float:
    de_score = normalize_debt_to_equity(metrics["debt_to_equity"])
    margin_score = normalize_profit_margin(metrics["profit_margin"])
    fcf_score = normalize_free_cash_flow(metrics["free_cash_flow_crores"])

    weighted = (
        de_score    * WEIGHT_DEBT_TO_EQUITY +
        margin_score * WEIGHT_PROFIT_MARGIN +
        fcf_score   * WEIGHT_FREE_CASH_FLOW
    )
    return round(weighted, 2)


# ── main entry point ───────────────────────────────────────────────────────────

def run_agent_alpha(ticker: str) -> dict:
    """
    Entry point called by the orchestrator.
    Accepts a ticker string like "RELIANCE.NS".
    Always returns a dict matching the output contract — never raises.
    """
    pdf_path = os.path.join(PDF_FOLDER, f"{ticker}_earnings.pdf")

    try:
        raw_text = extract_text_from_pdf(pdf_path)
        metrics = query_llm_for_metrics(raw_text)
        final_score = compute_weighted_score(metrics)

        return {
            "agent_id": "alpha_fundamental",
            "status": "success",
            "normalized_score": final_score,
            "raw_metrics": {
                "debt_to_equity": metrics["debt_to_equity"],
                "profit_margin": metrics["profit_margin"],
                "free_cash_flow_crores": metrics["free_cash_flow_crores"],
                "key_driver": metrics["key_driver"],
            },
        }

    except Exception as error:
        return {
            "agent_id": "alpha_fundamental",
            "status": "fallback",
            "normalized_score": 5.0,
            "raw_metrics": {
                "key_driver": f"Agent Alpha failed: {error}",
            },
        }