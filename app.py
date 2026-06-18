"""
app.py — Fact Check Agent

Pipeline: PDF -> extract text -> extract claims -> web search ->
LLM verification -> results table + downloadable CSV report.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from utils.pdf_reader import extract_text
from utils.claim_extractor import extract_claims
from utils.verifier import get_tavily_client, search_claim, verify_claim

load_dotenv()  # loads .env for local development


def get_secret(key: str) -> str:
    """
    Streamlit Cloud injects secrets via st.secrets; locally we fall
    back to the .env file. This lets the same code run in both places.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


st.set_page_config(page_title="Fact Check Agent", page_icon="✅", layout="wide")
st.title("📄 Fact Check Agent")
st.caption("Upload a PDF — claims get extracted and checked against live web data.")

openai_key = get_secret("OPENAI_API_KEY")
tavily_key = get_secret("TAVILY_API_KEY")

if not openai_key or not tavily_key:
    st.error(
        "Missing API keys. Add OPENAI_API_KEY and TAVILY_API_KEY to your "
        ".env file (local) or to Streamlit secrets (deployed)."
    )
    st.stop()

openai_client = OpenAI(api_key=openai_key)
tavily_client = get_tavily_client()

pdf = st.file_uploader("Upload a PDF", type=["pdf"])
max_claims = st.slider(
    "Max claims to check", min_value=3, max_value=20, value=10,
    help="Caps runtime and API cost — raise it if the PDF is dense with stats.",
)

if pdf:
    with st.spinner("Reading PDF..."):
        try:
            text = extract_text(pdf)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    with st.spinner("Extracting claims..."):
        claims = extract_claims(text, openai_client, max_claims=max_claims)

    if not claims:
        st.warning("No checkable factual claims were found in this document.")
        st.stop()

    st.success(f"Found {len(claims)} claim(s) to verify.")

    results = []
    progress = st.progress(0.0)

    for i, claim in enumerate(claims):
        with st.spinner(f"Checking claim {i + 1}/{len(claims)}..."):
            web_results = search_claim(claim, tavily_client)
            verdict = verify_claim(claim, web_results, openai_client)
            results.append(verdict)
        progress.progress((i + 1) / len(claims))

    df = pd.DataFrame(results)[["claim", "verdict", "correct_fact", "reason", "sources"]]
    df.columns = ["Claim", "Verdict", "Correct Fact", "Reason", "Sources"]

    def highlight_verdict(val):
        colors = {
            "Verified": "background-color: #d4edda",
            "Inaccurate": "background-color: #fff3cd",
            "False": "background-color: #f8d7da",
        }
        return colors.get(val, "")

    st.subheader("Results")
    st.dataframe(
        df.style.applymap(highlight_verdict, subset=["Verdict"]),
        use_container_width=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download report as CSV", csv, "fact_check_report.csv", "text/csv")