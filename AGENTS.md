# AGENTS.md

## Project goal
Analyze TTC bus reliability using Toronto open data.
Focus on headway instability, bunching, service gaps, and dashboard-ready outputs.

## Working rules
- Read the repo before coding.
- For non-trivial tasks, make a short plan first.
- Keep changes minimal and modular.
- Prefer reusable Python code in `src/` over notebook-only logic.
- Do not modify raw data in place.

## Data rules
- `data/raw/` = immutable source files
- `data/processed/` = cleaned outputs
- Document joins between schedule and realtime data
- Flag missing timestamps, duplicates, and route/stop mismatches

## Metrics to prioritize
- headway deviation
- bunching flag
- service gap flag
- route reliability summary

## Validation
After major changes, report:
- rows processed
- null summary
- duplicate summary
- sample output
- files changed

## Done means
A task is done only if:
1. code runs,
2. outputs are reproducible,
3. assumptions are documented,
4. results are summarized in plain English.
