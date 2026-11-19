"""Valeurs par defaut, scenario et chemins locaux."""

from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent


DB_PATH = HERE / "data" / "coastline.db"


OUT_DIR = HERE / "outputs" / "route_pdf"


DEPARTURE = {
    "code": "SGSIN", "name": "Singapore", "lat": 1.22521, "lon": 103.891,
    "time": datetime(2025, 8, 9, 14, 0, tzinfo=timezone.utc),
}


ARRIVAL = {
    "code": "CNTXG", "name": "Tianjin Xingang", "lat": 38.8475, "lon": 118.242,
    "time": datetime(2025, 8, 16, 5, 0, tzinfo=timezone.utc),
}


BBOX = (99.0, 127.0, -3.0, 42.0)


RES_DEG = 0.01


MARGIN_NM = 1.0


SOG_KN = 17.5


CMA_DISTANCE_NM = 2799.4
