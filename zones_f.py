"""Surfaces interdites, hypothese iceberg et masque F."""

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from shapely import LineString, Polygon, Point, box, intersects, prepare, set_precision, from_wkb, to_wkb
from shapely.ops import unary_union, polygonize
from .donnees import load_coast_geometries, read_area_rings
from .geometrie import lat_to_merc


ICEBERG_BBOX = (-66.0, -44.0, 44.0, 62.0)


ICEBERG_CONTINUATION_NEXT = (-52.833333333333336, 60.5)


GEOMETRY_PRECISION = 1e-8  # degre Mercator ; stabilisation numerique ~millimetrique


@dataclass
class ForbiddenAreas:
    surfaces: list              # liste de (nom, geometrie en Mercator)
    source_path: str
    iceberg_region_built: bool
    source_F_count: int = 0


def iceberg_cache_path(db_path, principal, western, north):
    """Cle dependante des geometries, de la base de cotes et de l'hypothese."""
    stat = Path(db_path).stat()
    fingerprint = hashlib.sha256()
    fingerprint.update(b'PSC-F-v1-mercator-3857-node-full-lines')
    fingerprint.update(np.asarray(principal).tobytes())
    fingerprint.update(np.asarray(western).tobytes())
    fingerprint.update(to_wkb(north))
    fingerprint.update(str((str(Path(db_path).resolve()), stat.st_size, stat.st_mtime_ns,
                            ICEBERG_BBOX, ICEBERG_CONTINUATION_NEXT, GEOMETRY_PRECISION)).encode())
    folder = Path(__file__).resolve().parent.parent / 'outputs' / 'cache_F'
    return folder / (fingerprint.hexdigest() + '.wkb')


def validate_iceberg_region(risk, frame):
    """Verifie le cote selectionne et l'independance du cadre artificiel."""
    if not risk.is_valid or risk.is_empty:
        raise ValueError('Surface glaciaire reconstruite invalide')
    if risk.distance(frame.boundary) < 10 * GEOMETRY_PRECISION:
        raise ValueError('Surface glaciaire non fermee : elle rejoint le bord artificiel du domaine')
    if not risk.contains(Point(-53.0, float(lat_to_merc(52.0)))):
        raise ValueError('Point de reference absent de la surface glaciaire')
    if not risk.contains(Point(-57.8, float(lat_to_merc(50.7)))):
        raise ValueError('Le cote est de la limite occidentale manque dans la surface')
    if risk.covers(Point(-59.7, float(lat_to_merc(50.0)))):
        raise ValueError('Le cote ouest du golfe est inclus a tort')


# Raccord ajoute selon l'hypothese du PSC ; ce point ne provient pas du CSV.
def build_iceberg_region(db_path, principal, western, north):
    """Ferme le cote a risque avec les cotes et le premier segment estime.

    Les intersections des lignes et surfaces sont nodees dans un meme plan
    a precision fixe pour eviter les micro-ouvertures numeriques.
    """
    lo, hi, la, ha = ICEBERG_BBOX
    frame = box(lo, float(lat_to_merc(la)), hi, float(lat_to_merc(ha)))
    cache = iceberg_cache_path(db_path, principal, western, north)
    if cache.exists():
        risk = from_wkb(cache.read_bytes())
        validate_iceberg_region(risk, frame)
        print('  Reconstruction F : surface deja verifiee rechargee.', flush=True)
        return risk
    print('  Reconstruction F : lecture et union des cotes...', flush=True)
    coast = unary_union(load_coast_geometries(db_path, ICEBERG_BBOX))
    print('  Reconstruction F : preparation du domaine maritime...', flush=True)
    ocean = set_precision(frame.difference(coast).difference(north), GEOMETRY_PRECISION)
    next_point = np.array([ICEBERG_CONTINUATION_NEXT[0], lat_to_merc(ICEBERG_CONTINUATION_NEXT[1])])
    if not north.contains(Point(next_point)):
        raise ValueError('Le raccord estime ne rejoint plus north water : verifier les donnees')
    extension = LineString([principal[-1], next_point])
    junctions = extension.intersection(north.boundary)
    if junctions.is_empty:
        raise ValueError('Intersection du raccord avec north water absente')
    main = set_precision(LineString(np.vstack([principal, next_point])), GEOMETRY_PRECISION)
    west = set_precision(LineString(western), GEOMETRY_PRECISION)
    # Garder les lignes entieres pendant le nodage. Les couper a l'ocean
    # avant l'union peut laisser des micro-ecarts aux raccordements aux cotes.
    network = unary_union([ocean.boundary, main, west])
    print('  Reconstruction F : selection du cote interdit...', flush=True)
    seed = Point(-53.0, float(lat_to_merc(52.0)))
    candidates = [p for p in polygonize(network) if p.contains(seed)]
    if len(candidates) != 1:
        raise ValueError('Impossible de selectionner une surface glaciaire unique')
    risk = candidates[0].intersection(ocean)
    validate_iceberg_region(risk, frame)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.tmp')
    temporary.write_bytes(to_wkb(risk))
    temporary.replace(cache)
    return risk


def load_forbidden_areas(area_path, db_path, grid):
    """Lit toutes les zones F ; les zones W ne sont pas rendues interdites."""
    csv.field_size_limit(10_000_000)
    with Path(area_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        required = {'CODE_AREA', 'NAME_AREA', 'RULES', 'COORD_AREA'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Colonnes requises absentes de area.csv')
        rows = [r for r in reader if r['CODE_AREA'].strip() == 'F']
    if not rows:
        raise ValueError('Aucune zone F dans area.csv')
    closed, lines = {}, {}
    for r in rows:
        name = r['NAME_AREA'].strip()
        if r['RULES'].strip().upper() != 'FORBIDDEN':
            raise ValueError(f'Regle F non prise en charge pour {name}')
        if name in closed or name in lines:
            raise ValueError(f'Nom de zone F duplique : {name}')
        rings = read_area_rings(r['COORD_AREA'])
        if name in ('ICEBERG LIMIT', 'WESTERN ICEBERG LIMIT'):
            if len(rings) != 1 or len(rings[0]) < 2:
                raise ValueError(f'Limite ouverte invalide : {name}')
            lines[name] = rings[0]
        else:
            if any(len(p) < 4 or not np.array_equal(p[0], p[-1]) for p in rings):
                raise ValueError(f'Contour F ouvert non interprete : {name}')
            p = Polygon(rings[0], rings[1:])
            if not p.is_valid:
                raise ValueError(f'Surface F invalide : {name}')
            closed[name] = p
    surfaces = list(closed.items())
    built = False
    if lines:
        if set(lines) != {'ICEBERG LIMIT', 'WESTERN ICEBERG LIMIT'} or 'north water' not in closed:
            raise ValueError('Les deux limites et north water sont necessaires a la reconstruction')
        lo, hi, la, ha = ICEBERG_BBOX
        frame = box(lo, float(lat_to_merc(la)), hi, float(lat_to_merc(ha)))
        extent = box(grid.lon0, grid.y_top-grid.n_rows*grid.res,
                     grid.lon0+grid.n_cols*grid.res, grid.y_top)
        if extent.intersects(frame):
            # Verifier les sommets relies par le segment retrouve : ne pas
            # appliquer aveuglement l'hypothese a une limite modifiee.
            expected = np.array([-55.75, lat_to_merc(59.0)])
            if not np.allclose(lines['ICEBERG LIMIT'][-1], expected, rtol=0, atol=1e-7):
                raise ValueError('Extremite iceberg modifiee : raccord a redefinir')
            risk = build_iceberg_region(db_path, lines['ICEBERG LIMIT'],
                                        lines['WESTERN ICEBERG LIMIT'], closed['north water'])
            surfaces.append(('icebergs_atlantique_raccord_hypothese', risk))
            built = True
    return ForbiddenAreas(surfaces, str(Path(area_path).resolve()), built, len(rows))


def forbidden_mask(grid, areas, label="F"):
    """Toutes les cellules touchant une surface F, sur LA grille du calcul.

    Calcul par bandes pour limiter la memoire. Les contacts avec le bord
    sont interdits aussi ; les trous des polygones sont respectes.
    """
    key = hashlib.sha256(b'PSC-F-mask-v1-cell-intersects')
    key.update(str((grid.lon0, grid.y_top, grid.res, grid.n_rows, grid.n_cols)).encode())
    for name, geometry in areas.surfaces:
        key.update(name.encode()); key.update(to_wkb(geometry))
    cache = Path(__file__).resolve().parent.parent / 'outputs' / 'cache_F' / ('mask-' + key.hexdigest() + '.npz')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            result = saved['masque_F']
        if result.shape != (grid.n_rows, grid.n_cols) or result.dtype != bool:
            raise ValueError('Masque F enregistre incompatible avec la grille')
        return result
    result = np.zeros((grid.n_rows, grid.n_cols), dtype=bool)
    for name, geometry in areas.surfaces:
        xmin, ymin, xmax, ymax = geometry.bounds
        c0 = max(0, math.floor((xmin-grid.lon0)/grid.res)-1)
        c1 = min(grid.n_cols, math.ceil((xmax-grid.lon0)/grid.res)+1)
        r0 = max(0, math.floor((grid.y_top-ymax)/grid.res)-1)
        r1 = min(grid.n_rows, math.ceil((grid.y_top-ymin)/grid.res)+1)
        if c0 >= c1 or r0 >= r1:
            continue
        print(f'  Cellules {label} : {name}', flush=True)
        prepare(geometry)
        cols = np.arange(c0, c1)
        left = grid.lon0 + cols*grid.res
        right = grid.lon0 + (cols+1)*grid.res
        for row in range(r0, r1, 32):
            rows = np.arange(row, min(row+32, r1))
            top = grid.y_top - rows*grid.res
            bottom = grid.y_top - (rows+1)*grid.res
            cells = box(left[None, :], bottom[:, None], right[None, :], top[:, None])
            result[row:row+len(rows), c0:c1] |= intersects(geometry, cells)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, masque_F=result)
    return result
