"""Lecture seule des cotes et des coordonnees area.csv."""

import re
import sqlite3
import struct
from pathlib import Path
import numpy as np
from shapely import Polygon
from .geometrie import lat_to_merc


SPATIALITE_POLYGON = 3


def parse_spatialite_polygon(blob: bytes) -> list[np.ndarray]:
    """Blob Spatialite -> anneaux de couples (longitude, latitude)."""
    if blob[0] != 0x00 or blob[38] != 0x7C or blob[-1] != 0xFE:
        raise ValueError("blob Spatialite invalide")
    endian = "<" if blob[1] == 1 else ">"
    (geom_type,) = struct.unpack(endian + "i", blob[39:43])
    if geom_type != SPATIALITE_POLYGON:
        raise ValueError(f"type de geometrie non gere : {geom_type}")
    (n_rings,) = struct.unpack(endian + "i", blob[43:47])
    offset = 47
    rings = []
    for _ in range(n_rings):
        (n_points,) = struct.unpack(endian + "i", blob[offset:offset + 4])
        offset += 4
        coords = np.frombuffer(blob, dtype=endian + "f8", count=2 * n_points, offset=offset)
        rings.append(coords.reshape(n_points, 2).copy())
        offset += 16 * n_points
    return rings


def load_land_polygons(db_path: str, bbox: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Charge les anneaux exterieurs de terre intersectant bbox via l'index R-tree.

    bbox = (lon_min, lon_max, lat_min, lat_max).
    Comme dans le PDF, seul le premier anneau de chaque polygone est conserve.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    query = """
        SELECT m.geom FROM map AS m
        JOIN idx_map_geom AS i ON i.pkid = m.rowid
        WHERE i.xmax >= ? AND i.xmin <= ? AND i.ymax >= ? AND i.ymin <= ?
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(query, (lon_min, lon_max, lat_min, lat_max)).fetchall()
    finally:
        con.close()
    return [parse_spatialite_polygon(blob)[0] for (blob,) in rows]


def read_area_rings(text):
    """Lit les couples du POLYGON du CSV, meme pour les limites ouvertes.

    Cette fonction ne ferme jamais un contour automatiquement.
    """
    if not re.fullmatch(r'\s*POLYGON\s*\(\s*\(.*\)\s*\)\s*', text):
        raise ValueError('Geometrie area.csv non prise en charge : POLYGON attendu')
    interior = text[text.index('(')+1:text.rindex(')')]
    rings = []
    for part in re.findall(r'\(([^()]*)\)', interior):
        p = np.array([tuple(map(float, pair.split())) for pair in part.split(',')])
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('Coordonnees invalides dans area.csv')
        # EPSG:3857 (metres) -> Mercator en degres equivalents du routeur.
        rings.append(np.degrees(p / 6_378_137.0))
    if not rings:
        raise ValueError('Geometrie vide dans area.csv')
    return rings


def load_coast_geometries(db_path, bbox):
    """Lecture de toutes les bagues, y compris les trous, pour la reconstruction F."""
    lo, hi, la, ha = bbox
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        rows = con.execute(
            'SELECT m.geom FROM map m JOIN idx_map_geom i ON i.pkid=m.rowid '
            'WHERE i.xmax>=? AND i.xmin<=? AND i.ymax>=? AND i.ymin<=?',
            (lo, hi, la, ha)).fetchall()
    finally:
        con.close()
    result = []
    for (blob,) in rows:
        rings = parse_spatialite_polygon(blob)
        rings = [np.column_stack([r[:, 0], lat_to_merc(r[:, 1])]) for r in rings]
        p = Polygon(rings[0], rings[1:])
        if not p.is_valid:
            raise ValueError('Polygone de cote invalide : reconstruction F interrompue')
        result.append(p)
    return result
