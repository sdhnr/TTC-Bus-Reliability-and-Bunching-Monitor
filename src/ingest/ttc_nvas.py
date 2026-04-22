"""Minimal TTC NVAS ingestion helpers (milestone 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

DATASET_URL = "https://open.toronto.ca/dataset/ttc-bustime-real-time-next-vehicle-arrival-nvas/"
GTFS_RT_BASE_URL = "https://bustime.ttc.ca/gtfsrt/"


def get_nvas_source_info() -> Dict[str, str]:
    """Return a short, beginner-friendly description of the NVAS access method."""
    return {
        "dataset_url": DATASET_URL,
        "access_method": "GTFS-Realtime web feed endpoints",
        "format": "GTFS-RT (Protocol Buffers); this milestone saves a small tabular sample CSV",
        "vehicle_positions_endpoint": f"{GTFS_RT_BASE_URL}vehicles",
    }


def _fallback_sample_rows(max_records: int) -> List[dict]:
    """Small fallback sample when network/protobuf parsing is not available."""
    rows = [
        {
            "vehicle_id": "TTC_DEMO_1001",
            "route_id": "29",
            "trip_id": "DEMO_TRIP_1",
            "latitude": 43.7001,
            "longitude": -79.4163,
            "timestamp_utc": "2026-04-22T12:00:00Z",
            "current_status": "IN_TRANSIT_TO",
            "source_note": "fallback_sample",
        },
        {
            "vehicle_id": "TTC_DEMO_1002",
            "route_id": "36",
            "trip_id": "DEMO_TRIP_2",
            "latitude": 43.7082,
            "longitude": -79.5181,
            "timestamp_utc": "2026-04-22T12:01:00Z",
            "current_status": "STOPPED_AT",
            "source_note": "fallback_sample",
        },
        {
            "vehicle_id": "TTC_DEMO_1003",
            "route_id": "39",
            "trip_id": "DEMO_TRIP_3",
            "latitude": 43.7738,
            "longitude": -79.3355,
            "timestamp_utc": "2026-04-22T12:02:00Z",
            "current_status": "IN_TRANSIT_TO",
            "source_note": "fallback_sample",
        },
    ]
    return rows[:max_records]


def download_or_create_nvas_sample(
    output_path: str | Path = "data/raw/ttc_nvas_vehicle_positions_sample.csv",
    max_records: int = 100,
) -> Dict[str, object]:
    """Save a small NVAS sample into data/raw as CSV.

    Tries a simple JSON read from the vehicles endpoint first; if unavailable,
    writes a small fallback sample to keep milestone-1 reproducible.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    note = "fallback_sample"

    try:
        request = Request(
            f"{GTFS_RT_BASE_URL}vehicles?format=json",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")

        data = json.loads(payload)
        entities = data.get("entity", []) if isinstance(data, dict) else []

        for entity in entities[:max_records]:
            vehicle = entity.get("vehicle", {}) if isinstance(entity, dict) else {}
            trip = vehicle.get("trip", {}) if isinstance(vehicle, dict) else {}
            position = vehicle.get("position", {}) if isinstance(vehicle, dict) else {}
            descriptor = vehicle.get("vehicle", {}) if isinstance(vehicle, dict) else {}

            rows.append(
                {
                    "vehicle_id": descriptor.get("id"),
                    "route_id": trip.get("route_id"),
                    "trip_id": trip.get("trip_id"),
                    "latitude": position.get("latitude"),
                    "longitude": position.get("longitude"),
                    "timestamp_utc": vehicle.get("timestamp"),
                    "current_status": vehicle.get("current_status"),
                    "source_note": "live_json_endpoint",
                }
            )

        if rows:
            note = "live_json_endpoint"
        else:
            rows = _fallback_sample_rows(max_records=max_records)

    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        rows = _fallback_sample_rows(max_records=max_records)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return {
        "saved_path": str(output_path),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "source_note": note,
    }
