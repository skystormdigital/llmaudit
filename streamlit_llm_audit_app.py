"""
streamlit_llm_audit_app.py  –  Lightweight LLM Visibility Auditor (Free Edition)
Author: ChatGPT scaffold • July 1 2025
Tested on: Python 3.10, Streamlit 1.35 (free tier)

This MVP avoids paid APIs by using:
• requests & trafilatura for crawling + text extraction
• sentence‑transformers (all‑MiniLM‑L6‑v2) for embeddings
• scikit‑learn MiniBatchKMeans for clustering
• BeautifulSoup & regex‑based heuristics for schema detection

Limitations (v0.1):
• Embeddings & clustering run in‑process ⇒ keep crawl ≤50 pages
• No generative LLMs (OpenAI, Claude, etc.) – only heuristic scoring
• CWV & lighthouse metrics not yet integrated (future work)
"""

from __future__ import annotations

import streamlit as st
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from datetime import datetime
import re

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load SBERT model once per session"""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def fetch_text(url: str, timeout: int = 10) -> str:
    """Pull a URL & return main article text (utf‑8)."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        # Use trafilatura to extract readable portion
        extracted = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=False,
            favour_precision=True,
        )
        return extracted or ""
    except Exception:
        return ""


def discover_urls(start_url: str, max_pages: int = 20) -> list[str]:
    """Breadth‑first crawl within the same domain up to *max_pages* URLs."""
    queue: list[str] = [start_url]
    visited: set[str] = set()
    discovered: list[str] = []
    domain = urlparse(start_url).netloc

    while queue and len(discovered) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            continue
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            continue

        discovered.append(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]  # strip fragments & query
            abs_url = urljoin(url, href)
            if (
                urlparse(abs_url).netloc == domain
                and abs_url not in visited
                and abs_url.startswith("http")
            ):
                queue.append(abs_url)

    return discovered


def embed_pages(urls: list[str], embedder: SentenceTransformer):
    texts = [fetch_text(u) for u in urls]
    st.info("Embedding pages… this may take a minute for larger sites.")
    embeddings = embedder.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    return texts, embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5):
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings) // 2 or 1)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels


# ---------------- Heuristic scores ----------------

def schema_score(html: str) -> float:
    """Binary 1/0 if Schema.org JSON‑LD present."""
    if re.search(r'"@context".*?"schema\\.org"', html, re.I | re.S):
        return 1.0
    return 0.0


def freshness_score(headers: dict) -> float:
    """0‑1 based on Last‑Modified recency."""
    lm = headers.get("Last-Modified")
    if not lm:
        return 0.5
    try:
        dt = datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z")
        age_days = (datetime.utcnow() - dt).days
    except Exception:
        return 0.5

    if age_days <= 30:
        return 1.0
    if age_days <= 365:
        return 0.7
    return 0.3


def keyword_cover_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return hits / len(keywords)


def heuristic_audit(urls: list[str], texts: list[str], keywords: list[str]) -> pd.DataFrame:
    rows = []
    for url, text in zip(urls, texts):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            headers = r.headers
            html = r.text
        except Exception:
            headers, html = {}, ""

        rows.append(
            {
                "url": url,
                "word_count": len(text.split()),
                "schema": schema_score(html),
                "freshness": freshness_score(headers),
                "keyword_coverage": keyword_cover_score(text, keywords),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

def main():
    st.set_page_config(page_title="LLM Visibility Auditor", layout="wide")
    st.title("🔍 LLM Visibility Auditor (Free Edition)")

    with st.sidebar:
        domain = st.text_input("Start URL (e.g., https://example.com)")
        max_pages = st.slider("Max pages to crawl", 5, 50, 20, 1)
        kw_input = st.text_area("Optional: comma‑separated keywords/intents")
        run_button = st.button("Run audit ▶️")

    if run_button and domain:
        with st.spinner("Crawling site…"):
            urls = discover_urls(domain, max_pages)
        st.success(f"Discovered {len(urls)} pages in the same domain.")

        embedder = load_embedder()
        texts, embeddings = embed_pages(urls, embedder)
        labels = cluster_embeddings(np.array(embeddings))

        kw_list = [k.strip() for k in kw_input.split(",") if k.strip()]
        df = heuristic_audit(urls, texts, kw_list)
        df["cluster"] = labels

        st.subheader("📄 Page‑level audit results")
        st.dataframe(df, height=500)

        csv = df.to_csv(index=False).encode("utf‑8")
        st.download_button("Download CSV", csv, "llm_audit.csv", "text/csv")

        # Quick insights
        st.subheader("💡 Quick insights")
        st.write(f"**Schema coverage:** {df['schema'].mean()*100:.1f}% of pages include Schema.org JSON‑LD.")
        if df['freshness'].mean() < 0.6:
            st.warning("Many pages appear stale – consider updating high‑value content.")
        if kw_list and df['keyword_coverage'].mean() < 0.5:
            st.warning("Keyword gaps detected – several intents missing from content.")

        st.markdown("---")
        st.caption("Made with ❤️   |   Libraries: Streamlit · Sentence‑Transformers · Trafilatura · scikit‑learn · BeautifulSoup 4")


if __name__ == "__main__":
    main()
