import streamlit as st
import json
import time

# --- 1. THE MOCK DATA CONTRACT ---
# This simulates the exact JSON your backend partner will eventually send you.
# Until they finish, you just read from this string.
mock_data = """
{
  "ticker": "RELIANCE.NS",
  "timestamp": "2026-03-29T10:15:30Z",
  "final_decision": {
    "action": "STRONG BUY",
    "confidence_pct": 78.4,
    "suggested_allocation_pct": 5.0
  },
  "explainability_breakdown": {
    "fundamental_agent": {
      "score": 8.5,
      "key_driver": "Strong Free Cash Flow and low Debt-to-Equity."
    },
    "technical_agent": {
      "score": 4.2,
      "key_driver": "Approaching oversold territory (RSI 35)."
    },
    "sentiment_agent": {
      "score": 9.1,
      "key_driver": "Highly positive news volume regarding new green energy contracts."
    }
  }
}
"""

# Parse the mock data into a Python dictionary
data = json.loads(mock_data)

# --- 2. THE UI LAYOUT ---
st.set_page_config(page_title="AI Quant Terminal", layout="wide")

st.title("⚡ Multi-Agent AI Quant Terminal")
st.markdown("Institutional-grade analysis powered by distributed AI agents.")
st.divider()

# Search Bar
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter NSE Stock Ticker", value="RELIANCE.NS")
with col2:
    st.write("") # Spacing
    analyze_button = st.button("Run AI Agents", use_container_width=True)

# --- 3. THE INTERACTION ---
if analyze_button:
    # Fake loading state to make the demo look realistic
    with st.spinner("Agent Alpha reading quarterly PDFs..."):
        time.sleep(1)
    with st.spinner("Agent Beta calculating RSI and MACD..."):
        time.sleep(1)
    with st.spinner("Master Node crunching final XGBoost weights..."):
        time.sleep(1)
        
    st.success(f"Analysis complete for {ticker_input}")
    
    # --- 4. THE DASHBOARD ---
    # Top Row: The Master Node Final Decision
    st.subheader("Master Node Verdict")
    m1, m2, m3 = st.columns(3)
    
    # Change color based on Buy/Sell
    action_color = "normal" if data['final_decision']['action'] == "HOLD" else "inverse"
    
    m1.metric(label="Action", value=data['final_decision']['action'], delta="High Conviction")
    m2.metric(label="AI Confidence", value=f"{data['final_decision']['confidence_pct']}%")
    m3.metric(label="Suggested Portfolio Weight", value=f"{data['final_decision']['suggested_allocation_pct']}%")
    
    st.divider()
    
    # Bottom Row: The Explainability (The 3 Agents)
    st.subheader("Agent Explainability Breakdown")
    a1, a2, a3 = st.columns(3)
    
    with a1:
        st.info("📊 Agent Alpha: Fundamentals")
        st.metric(label="Score", value=f"{data['explainability_breakdown']['fundamental_agent']['score']} / 10")
        st.caption(data['explainability_breakdown']['fundamental_agent']['key_driver'])
        
    with a2:
        st.warning("📈 Agent Beta: Technicals")
        st.metric(label="Score", value=f"{data['explainability_breakdown']['technical_agent']['score']} / 10")
        st.caption(data['explainability_breakdown']['technical_agent']['key_driver'])
        
    with a3:
        st.success("📰 Agent Gamma: Sentiment")
        st.metric(label="Score", value=f"{data['explainability_breakdown']['sentiment_agent']['score']} / 10")
        st.caption(data['explainability_breakdown']['sentiment_agent']['key_driver'])