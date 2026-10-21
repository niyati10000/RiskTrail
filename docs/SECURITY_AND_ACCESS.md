# SECURITY_AND_ACCESS.md — Data, Keys, Hosting & Responsible Use

This is an ML project, so "security" mostly means data governance, secrets hygiene, reproducibility, and responsible lending use.

## 1. Data governance
- Use only data you are licensed to use (competition terms, university-provided, or synthetic). **Read the dataset's license/rules before sharing anything derived from it.**
- Competition data is typically **not redistributable**: never commit `data/raw/` or processed customer-level data to GitHub. `.gitignore` must include `data/`, `results/**/preds_*.npy` if they contain customer IDs, and any `.env`.
- No real PII. If any identifier is present, hash it and never display it in the demo.
- Keep a `DATA_NOTES.md` (or section in README) stating source, license, and date downloaded.

## 2. Secrets & API keys
| Secret | Where it lives | Notes |
|---|---|---|
| Kaggle API token (if downloading via CLI) | `~/.kaggle/kaggle.json` | never in repo; chmod 600 |
| Hugging Face token (deploy) | env var `HF_TOKEN` or HF CLI login | write-scoped only when deploying |
| GitHub token | git credential manager / SSH key | do not paste in prompts to Antigravity |
- Use a `.env` (gitignored) + `.env.example` with placeholder names.
- **Never paste real keys into AI prompts or brain files.** If a key leaks, revoke and regenerate immediately.
- Run `git log -p | grep -i token` style check or use `gitleaks` before the first public push.

## 3. Reproducibility & integrity
- Pin dependencies (`requirements.txt` with versions).
- Fix seeds (python, numpy, torch) in `utils/seed.py`; record seed + git commit hash in every `config.json`.
- Splits are saved once (`splits.parquet`) and reused; never re-split silently.
- Tests guard against leakage (customer in two splits, label used as a feature, future statements in features for time-aware check).

## 4. Hosting (optional demo)
- **Hugging Face Spaces (Streamlit)** is the simplest free option; code on GitHub, Space synced from it.
- Demo ships only a **small synthetic or anonymized sample**, plus the trained model files, not the full dataset.
- Cap upload size, no arbitrary file execution, no pickle loading from user uploads (load only your own saved models; prefer `torch.save(state_dict)` and LightGBM text models).
- No logging of user-entered values beyond what is needed.

## 5. Responsible use (lending context)
- State clearly in README and demo: **educational project, not for real credit decisions.**
- Fairness: no protected attributes should be used as features; if the dataset has proxies, note them as a limitation. Optionally compare error rates across available groups.
- Explanations: provide reason codes (SHAP top features) for declines, since adverse-action explanations are expected in real lending.
- Calibration matters: report it, because decisions and pricing depend on probabilities.
- Document the profit/loss assumptions as **assumptions**, not facts.

## 6. Pre-publish checklist
- [ ] No raw or customer-level data in repo
- [ ] No keys in code, notebooks, or git history
- [ ] LICENSE chosen for your code; dataset license respected
- [ ] README has reproduction steps and the educational-use disclaimer
- [ ] Demo uses sample/synthetic data only
- [ ] Notebook outputs cleared of any customer-level rows
