"""Zones traversables a vitesse plafonnee et profil final."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from shapely import LineString, Polygon, Point
from .donnees import read_area_rings
from .geometrie import lat_to_merc, merc_to_lat, rhumb_nm
from .zones_f import ForbiddenAreas, forbidden_mask


@dataclass
class SpeedArea:
    name: str
    geometry: object
    limit_kn: float


def load_speed_areas(area_path, unit='kn'):
    """Lecture stricte des W ; aucune fermeture implicite ni regle ignoree."""
    result = []
    csv.field_size_limit(10_000_000)
    factor = {'kn': 1., 'kmh': 1/1.852, 'mps': 3600/1852}[unit]
    with Path(area_path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream, delimiter=';'):
            if row['CODE_AREA'].strip() != 'W':
                continue
            match = re.fullmatch(r'CURRENT_SPEED\s*<=\s*([0-9]+(?:\.[0-9]+)?)', row['RULES'].strip())
            if not match or float(match[1]) <= 0:
                raise ValueError('Regle W inconnue ou invalide : '+row['RULES'])
            rings = read_area_rings(row['COORD_AREA'])
            if any(len(r)<4 or not np.allclose(r[0], r[-1], rtol=0, atol=1e-9) for r in rings):
                raise ValueError('Contour W ouvert : '+row['NAME_AREA'])
            geometry = Polygon(rings[0], rings[1:])
            if not geometry.is_valid or geometry.is_empty:
                raise ValueError('Surface W invalide : '+row['NAME_AREA'])
            result.append(SpeedArea(row['NAME_AREA'], geometry, float(match[1])*factor))
    return result


def speed_on_grid(grid, zones, cruise_kn):
    """Minimum des plafonds sur toute cellule touchee ; W reste navigable."""
    if not np.isfinite(cruise_kn) or cruise_kn <= 0:
        raise ValueError('Vitesse de croisiere strictement positive requise')
    speed = np.full((grid.n_rows, grid.n_cols), cruise_kn, dtype=np.float32)
    for zone in zones:
        if zone.limit_kn >= cruise_kn:
            continue
        # Reutilise le test cellule/surface F, sans ajouter le masque aux obstacles.
        mask = forbidden_mask(grid, ForbiddenAreas([(zone.name, zone.geometry)], '', False), label='W')
        speed[mask] = np.minimum(speed[mask], zone.limit_kn)
    return speed


def grid_segment_hours(grid, speed, p, q):
    """Quadrature sur le champ de vitesse rasterise, pour le lissage en temps."""
    n = max(2, int(np.ceil(4*np.max(np.abs(q-p))))+1)
    t = np.linspace(0., 1., n)
    rc = p[:, None] + (q-p)[:, None]*t
    rr = np.clip(np.rint(rc[0]).astype(int), 0, grid.n_rows-1)
    cc = np.clip(np.rint(rc[1]).astype(int), 0, grid.n_cols-1)
    lo, la = grid.to_lonlat(rc[0], rc[1])
    d = rhumb_nm(lo[:-1], la[:-1], lo[1:], la[1:])
    return float(np.sum(d*.5*(1/speed[rr[:-1],cc[:-1]] + 1/speed[rr[1:],cc[1:]])))


def route_speed_profile(lons, lats, zones, cruise_kn):
    """Decoupe EXACTE aux frontieres W en Mercator, puis distances loxodromiques.

    Chaque morceau porte sa vitesse autorisee ; les chevauchements prennent le
    minimum. Une tangence ponctuelle n'ajoute pas de duree. Aucun calendrier W.
    """
    pieces = []
    ys = lat_to_merc(np.asarray(lats))
    for leg in range(len(lons)-1):
        line = LineString([(lons[leg], ys[leg]), (lons[leg+1], ys[leg+1])])
        if line.length == 0:
            continue
        candidates = [z for z in zones if z.geometry.intersects(line)]
        cuts = [0., 1.]
        for zone in candidates:
            hit = line.intersection(zone.geometry.boundary)
            def vertices(g):
                if g.is_empty: return []
                if hasattr(g, 'geoms'): return [xy for part in g.geoms for xy in vertices(part)]
                return list(g.coords)
            cuts.extend(line.project(Point(xy), normalized=True) for xy in vertices(hit))
        cuts = sorted(set(cuts))
        for a,b in zip(cuts[:-1], cuts[1:]):
            if b-a < 1e-12: continue
            start, end = line.interpolate(a, normalized=True), line.interpolate(b, normalized=True)
            middle = line.interpolate((a+b)/2, normalized=True)
            active = [z for z in candidates if z.geometry.covers(middle)]
            limit = min([cruise_kn]+[z.limit_kn for z in active])
            lat1,lat2 = float(merc_to_lat(start.y)),float(merc_to_lat(end.y))
            distance = float(rhumb_nm(start.x, lat1, end.x, lat2))
            pieces.append(dict(leg=leg, lon_start=start.x, lat_start=lat1,
                lon_end=end.x, lat_end=lat2, distance_nm=distance, speed_kn=limit,
                duration_h=distance/limit, zones_W=[z.name for z in active]))
    return pieces
