import feedparser
from transformers import pipeline, AutoTokenizer
from optimum.pipelines import pipeline as onnx_pipeline

# ── constants ──────────────────────────────────────────────────────────────────

FINBERT_MODEL = "ProsusAI/finbert"
MAX_HEADLINES = 10
GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

# Ticker → human-readable search query for Google News
TICKER_TO_COMPANY_NAME = {
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "INFY.NS":      "Infosys",
    "HDFCBANK.NS":  "HDFC Bank",
    "WIPRO.NS":     "Wipro",
    "HINDUNILVR.NS":"Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS":      "State Bank of India",
}


# ── model loader (call once at startup, not per request) ──────────────────────

def load_finbert_onnx_pipeline():
    """
    Loads FinBERT in ONNX format via Hugging Face Optimum.
    ONNX runtime is ~250MB vs ~2GB for full PyTorch — no torch dependency.
    Call this once at app startup and pass the returned pipeline into run_agent_gamma().
    """
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    sentiment_classifier = onnx_pipeline(
        task="text-classification",
        model=FINBERT_MODEL,
        tokenizer=tokenizer,
        accelerator="ort",      # use ONNX Runtime instead of PyTorch
        top_k=None,             # return all three label probabilities
        truncation=True,
        max_length=512,
    )
    return sentiment_classifier


# ── stage 1: fetch headlines from google news rss ─────────────────────────────

def fetch_headlines(ticker: str) -> list[str]:
    company_name = TICKER_TO_COMPANY_NAME.get(
        ticker,
        ticker.replace(".NS", "").replace(".BSE", "")
    )
    rss_url = GOOGLE_NEWS_RSS_BASE.format(query=company_name.replace(" ", "+"))
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries[:MAX_HEADLINES]:
        title = entry.get("title", "").strip()
        # Google News RSS appends the source name after " - " — strip it
        # e.g. "Reliance Q3 profit rises 12% - Economic Times" → "Reliance Q3 profit rises 12%"
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        if title:
            headlines.append(title)

    return headlines


# ── stage 2: classify each headline with finbert onnx ─────────────────────────

def classify_headlines(headlines: list[str], sentiment_classifier) -> list[dict]:
    classified_results = []

    for headline in headlines:
        label_scores = sentiment_classifier(headline)[0]
        # label_scores is a list like:
        # [{"label": "positive", "score": 0.84}, {"label": "negative", "score": 0.08}, ...]
        scores_by_label = {item["label"]: round(item["score"], 4) for item in label_scores}

        classified_results.append({
            "headline": headline,
            "positive": scores_by_label.get("positive", 0.0),
            "negative": scores_by_label.get("negative", 0.0),
            "neutral":  scores_by_label.get("neutral",  0.0),
        })

    return classified_results


# ── stage 3: aggregate into a single score ────────────────────────────────────

def aggregate_sentiment_score(classified_headlines: list[dict]) -> float:
    if not classified_headlines:
        return 5.0

    headline_count = len(classified_headlines)
    average_positive = sum(item["positive"] for item in classified_headlines) / headline_count
    average_negative = sum(item["negative"] for item in classified_headlines) / headline_count

    # Net sentiment in range [-1, +1]
    net_sentiment = average_positive - average_negative

    # Map [-1, +1] → [0, 10]
    score = (net_sentiment + 1.0) / 2.0 * 10.0
    return round(min(10.0, max(0.0, score)), 2)


def build_key_driver(classified_headlines: list[dict]) -> str:
    if not classified_headlines:
        return "No headlines found for this ticker."

    most_positive = max(classified_headlines, key=lambda item: item["positive"])
    most_negative = max(classified_headlines, key=lambda item: item["negative"])

    positive_headline = most_positive["headline"][:80]
    negative_headline = most_negative["headline"][:80]

    return (
        f"Most positive: '{positive_headline}' "
        f"({round(most_positive['positive'] * 100)}% confidence). "
        f"Most negative: '{negative_headline}' "
        f"({round(most_negative['negative'] * 100)}% confidence)."
    )


# ── main entry point ───────────────────────────────────────────────────────────

def run_agent_gamma(ticker: str, sentiment_classifier) -> dict:
    """
    Entry point called by the orchestrator.
    Accepts a ticker string like "RELIANCE.NS" and the pre-loaded FinBERT pipeline.
    Always returns a dict matching the output contract — never raises.
    """
    try:
        headlines = fetch_headlines(ticker)
        classified = classify_headlines(headlines, sentiment_classifier)
        sentiment_score = aggregate_sentiment_score(classified)
        key_driver = build_key_driver(classified)

        headline_count = len(classified)
        average_positive = round(
            sum(item["positive"] for item in classified) / headline_count, 4
        ) if classified else 0.0
        average_negative = round(
            sum(item["negative"] for item in classified) / headline_count, 4
        ) if classified else 0.0

        return {
            "agent_id": "gamma_sentiment",
            "status": "success",
            "normalized_score": sentiment_score,
            "raw_metrics": {
                "headline_count": headline_count,
                "average_positive": average_positive,
                "average_negative": average_negative,
                "key_driver": key_driver,
            },
        }

    except Exception as error:
        return {
            "agent_id": "gamma_sentiment",
            "status": "fallback",
            "normalized_score": 5.0,
            "raw_metrics": {
                "key_driver": f"Agent Gamma failed: {error}",
            },
        }