# Methodology: zero-to-hero verification

1. **Sequential search, one at a time.** websearch (Exa) first; on 429 backoff 5s→10s (max 3); fallback webfetch on lite.duckduckgo. Then, each with distinct keywords: duckduckgo_search, openresearch (web/openalex/news/hacker_news/stackoverflow), paper-search (unified/arxiv/semantic/crossref/openalex), agent-reach_search (web), gitmcp (docs+code), kaggle (everything/discussions), wiki, gsd_websearch, superpowers semantic.
2. **Vote.** A claim enters the paper only with ≥2 independent sources; single-source figures stay labeled "author-reported".
3. **Compute before write.** Every table in README is produced by `experiments/01–04` and committed as `results/*.json`. The one post-hoc fix (03 cos threshold 0.9→0.3) is documented inline with the finite-sample reason — no silent edits.
4. **Live-data rule.** Public research needs public-data grounding: `04_live_hf_verify.py` hits HF Hub + GitHub + arXiv APIs (no token) on every run. Offline = soft-fail, never fake-pass.
5. **Units.** HF file sizes = decimal GB (1e9). GiB noted where llama.cpp uses it (~7% gap explains several "mismatches" online).
6. **Market/live check.** GPU VRAM tiers, download counts, star counts are point-in-time (2026-10-15) and re-checked per run; speeds are base-model runs, never attributed to untested builds.
