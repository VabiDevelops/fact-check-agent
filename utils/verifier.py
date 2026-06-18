"""
utils/verifier.py

Two responsibilities:
1. search_claim()  — hit Tavily for live web evidence on a claim.
2. verify_claim()  — ask the LLM to compare the claim against that
    evidence and return a structured verdict.

Key fixes over the original draft:
- verify_claim() always returns a parsed dict, never a raw string,
    so the UI doesn't have to guess what it got back.
- Strips markdown code fences before json.loads(), since LLMs often
    wrap JSON in ```json ... ``` even when told not to.
- If web search returns nothing, the claim is verdict "False" with
    a clear reason, instead of silently sending empty evidence to the
    LLM (which tends to hallucinate a verdict anyway).
- Wrapped in try/except so one bad claim doesn't crash the whole run.
"""

import os
import json
import re
from openai import OpenAI
from tavily import TavilyClient


def get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_claim(claim: str, tavily: TavilyClient, max_results: int = 5) -> list[dict]:
    """Returns a list of {title, url, content} dicts, or [] on failure/no results."""
    try:
        result = tavily.search(
            query=claim,
            search_depth="advanced",
            max_results=max_results,
        )
        return result.get("results", [])
    except Exception:
        return []


def verify_claim(claim: str, web_results: list[dict], llm: OpenAI) -> dict:
    """
    Compares a claim against retrieved web evidence and returns:
    {claim, verdict, correct_fact, reason, sources}
    """
    if not web_results:
        return {
            "claim": claim,
            "verdict": "False",
            "correct_fact": "N/A",
            "reason": "No supporting web evidence was found for this claim.",
            "sources": [],
        }

    evidence_blocks = []
    sources = []
    for r in web_results:
        url = r.get("url", "")
        content = r.get("content", "")
        evidence_blocks.append(f"Source: {url}\nContent: {content}")
        if url:
            sources.append(url)

    evidence = "\n\n".join(evidence_blocks)

    prompt = f"""
You are a professional fact checker. Compare the claim against the
web evidence below and decide a verdict.

Claim:
{claim}

Web Evidence:
{evidence}

Verdict rules:
- "Verified": evidence confirms the claim is accurate / current.
- "Inaccurate": evidence shows a different or updated figure/detail
(e.g. an outdated statistic, a number that has since changed).
- "False": evidence directly contradicts the claim.

Return ONLY valid JSON, no markdown fences, no extra commentary:
{{"verdict": "Verified" | "Inaccurate" | "False",
"correct_fact": "the accurate figure/fact, or 'N/A' if Verified",
"reason": "one sentence explaining the verdict"}}
"""

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
    except Exception as e:
        parsed = {
            "verdict": "Inaccurate",
            "correct_fact": "N/A",
            "reason": f"Could not parse verifier output ({e}).",
        }

    parsed["claim"] = claim
    parsed["sources"] = sources[:3]
    return parsed