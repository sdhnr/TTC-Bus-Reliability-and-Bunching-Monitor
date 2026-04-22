# TTC Bus Reliability and Bunching Monitor

This project analyzes TTC bus reliability using Toronto open data.

## Objective
Measure headway instability, detect bunching, identify service gaps, and prepare dashboard-ready outputs.

## Planned outputs
- cleaned transit datasets
- route and stop reliability metrics
- bunching hotspot analysis
- Power BI dashboard inputs
- short findings report

## TTC NVAS ingestion (milestone 1)
- **Dataset page:** https://open.toronto.ca/dataset/ttc-bustime-real-time-next-vehicle-arrival-nvas/
- **Access method assumption:** TTC BusTime GTFS-Realtime web feeds (for example `https://bustime.ttc.ca/gtfsrt/vehicles`).
- **Format assumption:** GTFS-RT feeds are Protocol Buffers. For this first beginner-friendly milestone, we save a small CSV sample in `data/raw/` for easy pandas exploration.
- **Environment note:** If live endpoint access is blocked, the ingestion script writes a tiny fallback sample so notebook steps stay reproducible.

## Phase 2: Modeling-ready analytics and prediction
The project now includes a reusable pipeline in `src/metrics/modeling_pipeline.py` to:
- audit available project assets,
- build a leakage-aware modeling table,
- define TTC bunching target,
- run time-aware split and model comparison,
- tune with `TimeSeriesSplit`,
- export dashboard-ready summaries and figures,
- optionally generate a stop hotspot map.

Primary phase-2 notebook: `notebooks/02_ttc_bunching_modeling.ipynb`.
