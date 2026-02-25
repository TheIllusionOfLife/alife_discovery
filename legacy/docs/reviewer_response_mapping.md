# Reviewer Response Mapping (Paper Final Polish)

This note maps high-priority reviewer concerns to concrete manuscript and tooling edits included in this PR.

## Concern-to-Edit Map

| Reviewer concern | Resolution | Location |
|---|---|---|
| Clarify "objective-free" vs implicit selection pressure | Reframed to "objective-free but viability-filtered"; explicit text that filters are viability constraints, not behavioral objectives | `paper/main.tex` (Abstract, Introduction, Related Work "Our position", Methods/Physical Filters, Conclusion) |
| Viability confound interpretation | Added explicit statement that post-filter metrics are never fed back into search and clarified interpretation as negative selection only | `paper/main.tex` (Methods/Physical Filters; Discussion/Viability filtering as minimal selection) |
| Ranking stability zero-overlap ambiguity | Added explicit interpretation that zero-overlap Kendall is non-identifiable and reported as N/A, not instability evidence | `paper/main.tex` (new subsection: Ranking-Stability Overlap Interpretation); `paper/supplementary.tex` (PR26 follow-up section) |
| Threats to validity: overlap/update-order | Expanded limitations to include overlap boundary-case interpretation and retained sequential update-order caveat | `paper/main.tex` (Limitations) |
| Reproducibility traceability | Added stable references (Zenodo DOI, git tag, baseline commit) and reviewer mapping pointer | `docs/reproducibility.md` |
| Artifact rendering semantics for undefined values | Standardized non-finite macro rendering to `N/A` (instead of `NaN`) | `scripts/render_pr26_followups_tex.py`; `paper/generated/pr26_followups.defaults.tex` |

## Validation Checklist

- `uv run ruff check .`
- `uv run ruff format . --check`
- `uv run pytest -q`
- `uv run python scripts/render_pr26_followups_tex.py --followup-dir data/post_hoc/pr26_followups --output paper/generated/pr26_followups.tex`
- `tectonic paper/main.tex`
- `tectonic paper/supplementary.tex`

