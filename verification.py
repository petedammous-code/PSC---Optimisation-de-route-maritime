"""Controles geometriques du trajet final."""

import numpy as np
from shapely import STRtree, LineString, Polygon
from .geometrie import lat_to_merc, merc_to_lat


def check_route_against_forbidden(areas, lons, lats):
    """Intersection vectorielle des segments finaux dans le plan Mercator."""
    violations = []
    y = lat_to_merc(np.asarray(lats))
    for k in range(len(lons)-1):
        segment = LineString([(lons[k], y[k]), (lons[k+1], y[k+1])])
        for name, geometry in areas.surfaces:
            if segment.intersects(geometry):
                violations.append({'segment': k, 'zone': name})
    return violations


def densify_rhumb(lon1, lat1, lon2, lat2, step_deg=0.005):
    """Points le long d'une loxodromie, droite dans le plan Mercator."""
    y1, y2 = lat_to_merc(lat1), lat_to_merc(lat2)
    n = int(np.ceil(max(abs(lon2 - lon1), abs(y2 - y1)) / step_deg)) + 1
    t = np.linspace(0, 1, n)
    return lon1 + t * (lon2 - lon1), merc_to_lat(y1 + t * (y2 - y1))


def check_route_against_land(polygons, lons, lats):
    polys = [Polygon(r) for r in polygons]
    tree = STRtree(polys)
    n_cross, min_clear = 0, np.inf
    for k in range(len(lons) - 1):
        lo, la = densify_rhumb(lons[k], lats[k], lons[k + 1], lats[k + 1])
        line = LineString(np.column_stack([lo, la]))
        if len(tree.query(line, predicate="intersects")):
            n_cross += 1
        # Projection equirectangulaire locale pour la distance approximative.
        cand = tree.query(line.buffer(0.3))
        if len(cand):
            c = np.cos(np.radians(np.mean(la)))
            line_loc = LineString(np.column_stack([lo * c, la]))
            for i in cand:
                r = polygons[i]
                d = Polygon(np.column_stack([r[:, 0] * c, r[:, 1]])).distance(line_loc) * 60
                min_clear = min(min_clear, d)
    return {"n_crossings": n_cross, "min_clearance_nm": float(min_clear)}
