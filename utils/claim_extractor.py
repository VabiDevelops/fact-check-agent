"""
utils/claim_extractor.py

Two-stage claim extraction:
  1. Cheap regex pass — keep only sentences that look like they might
     contain a checkable fact (numbers, %, currency, magnitude words).
     This is free and fast, and trims what we send to the LLM.
  2. One batched LLM call that turns those candidate sentences into
     clean, standalone, fact-checkable claims — dropping noise like
     page numbers, footnotes, or vague statements, and fixing
     fragments that lost their subject when split out of context.

This fixes the main weakness of pure regex extraction (your original
Step 5): it catches things like "Page 4 of 12" or footnote markers as
"claims," and it can't tell a real statistic from a stray number.
"""

import re
import json
from openai import OpenAI

CANDIDATE_PATTERN = re.compile(
    r"\d|%|\$|₹|€|£|\bpercent\b|\bmillion\b|\bbillion\b|\btrillion\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _prefilter_candidates(text: str) -> list[str]:
    sentences = _split_sentences(text)
    return [s for s in sentences if CANDIDATE_PATTERN.search(s)]


def extract_claims(text: str, client: OpenAI, max_claims: int = 10) -> list[str]:
    """
    Returns a list of clean, checkable claim strings (at most
    max_claims), pulled from the document via the two-stage process
    described above. Falls back to the raw regex candidates if the
    LLM step fails for any reason, so the app never breaks here.
    """
    candidates = _prefilter_candidates(text)
    if not candidates:
        return []

    # Cap input size sent to the LLM to control cost/latency on long PDFs.
    candidates = candidates[:60]

    prompt = f"""
You are helping a fact-checking tool prepare claims for verification.

Below are sentences pulled from a document. Some are genuine factual
claims (statistics, financial figures, dates, rankings, market sizes,
company metrics). Others are noise (page numbers, footnotes, headers,
or vague statements with no checkable fact).

Sentences:
{json.dumps(candidates, indent=2)}

Return ONLY a JSON array of strings, nothing else. Each string should
be one rewritten, self-contained, checkable claim (fix pronouns or
missing subjects so it can be understood on its own, with no
surrounding context needed). Drop anything that isn't a real
checkable claim. Return at most {max_claims} claims, prioritizing the
most specific and significant ones.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            cleaned = [c for c in claims if isinstance(c, str) and c.strip()]
            return cleaned[:max_claims]
    except json.JSONDecodeError:
        pass

    # Fallback: LLM step failed to return valid JSON — use raw candidates.
    return candidates[:max_claims]