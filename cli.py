"""Options de lancement, orchestration et exports."""

import argparse
import csv
import json
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
from shapely import Point
from shapely.geometry import mapping
from .cartes import plot_route
from .configuration import ARRIVAL, BBOX, CMA_DISTANCE_NM, DB_PATH, DEPARTURE, MARGIN_NM, OUT_DIR, RES_DEG, SOG_KN
from .donnees import load_land_polygons
from .geometrie import haversine_nm, lat_to_merc, path_length_nm, rhumb_nm
from .grille import build_grid
from .recherche import grid_shortest_path, line_of_sight, relax_waypoints, snap_to_navigable, string_pull
from .verification import check_route_against_forbidden, check_route_against_land
from .zones_f import forbidden_mask, load_forbidden_areas
from .zones_w import grid_segment_hours, load_speed_areas, route_speed_profile, speed_on_grid


def main():
    global RES_DEG, MARGIN_NM
    parser = argparse.ArgumentParser(description="Routage maritime avec zones F et limitations de vitesse W")
    parser.add_argument("--res", type=float, default=RES_DEG, help="pas de grille en degres (defaut 0.01)")
    parser.add_argument("--marge", type=float, default=MARGIN_NM, help="distance minimale a la cote en nm (defaut 1.0)")
    parser.add_argument("--no-plot", action="store_true", help="ne pas generer la carte")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="chemin de coastline.db")
    parser.add_argument("--output", type=Path, default=OUT_DIR, help="dossier des resultats")
    parser.add_argument('--area', type=Path, help='area.csv ; par defaut a cote de coastline.db')
    parser.add_argument('--sans-F', action='store_true', help='desactiver explicitement les zones F pour comparaison')
    parser.add_argument('--voisins', type=int, choices=(4, 8), default=8, help='voisins de Dijkstra (8 par defaut)')
    parser.add_argument('--bbox', nargs=4, type=float, default=BBOX, metavar=('OUEST', 'EST', 'SUD', 'NORD'))
    parser.add_argument('--depart', nargs=2, type=float, metavar=('LONGITUDE', 'LATITUDE'), help='point de demonstration')
    parser.add_argument('--arrivee', nargs=2, type=float, metavar=('LONGITUDE', 'LATITUDE'), help='point de demonstration')
    parser.add_argument('--sans-W', action='store_true', help='ignorer les limitations W pour comparaison')
    parser.add_argument('--vitesse', type=float, default=SOG_KN, help='vitesse de croisiere en noeuds')
    parser.add_argument('--unite-W', choices=('kn','kmh','mps'), default='kn', help='hypothese unite CURRENT_SPEED, kn par defaut')
    args = parser.parse_args()
    if not np.isfinite(args.vitesse) or args.vitesse <= 0:
        parser.error('--vitesse doit etre positive et finie')
    RES_DEG, MARGIN_NM = args.res, args.marge
    out_dir = args.output
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    departure, arrival = dict(DEPARTURE), dict(ARRIVAL)
    if args.depart:
        departure.update(lon=args.depart[0], lat=args.depart[1], code='DEPART', name='Point de demonstration')
    if args.arrivee:
        arrival.update(lon=args.arrivee[0], lat=args.arrivee[1], code='ARRIVEE', name='Point de demonstration')
    route_tag = departure['code'] + '_' + arrival['code']

    polygons = load_land_polygons(str(args.db.resolve()), tuple(args.bbox))
    print(f"[{time.time()-t0:5.1f}s] {len(polygons)} polygones de terre charges", flush=True)

    grid = build_grid(polygons, tuple(args.bbox), RES_DEG)
    nav_cotes = grid.navigable(MARGIN_NM)
    areas = None
    masque_F = np.zeros(nav_cotes.shape, dtype=bool)
    if not args.sans_F:
        area_path = args.area or args.db.resolve().parent / 'area.csv'
        areas = load_forbidden_areas(area_path, args.db, grid)
        masque_F = forbidden_mask(grid, areas)
        print(f'[{time.time()-t0:5.1f}s] zones F : {masque_F.sum()} cellules, raccord atlantique : {areas.iceberg_region_built}', flush=True)
    nav = nav_cotes & ~masque_F
    print(f"[{time.time()-t0:5.1f}s] grille {grid.n_rows} x {grid.n_cols} ({nav.mean():.0%} navigable)", flush=True)

    area_path = args.area or args.db.resolve().parent / 'area.csv'
    speed_areas = [] if args.sans_W else load_speed_areas(area_path, args.unite_W)
    speed = speed_on_grid(grid, speed_areas, args.vitesse)
    slowed_cells = int(np.count_nonzero((speed < args.vitesse - 1e-6*args.vitesse) & nav))
    time_objective = slowed_cells > 0
    print(f'Zones W : {len(speed_areas)} lues, {slowed_cells} cellules navigables ralenties ; unite supposee : {args.unite_W}', flush=True)
    endpoints = []
    for label, port in [('depart', departure), ('arrivee', arrival)]:
        rc = tuple(np.round(grid.to_rc(port['lon'], port['lat'])).astype(int))
        if not (0 <= rc[0] < grid.n_rows and 0 <= rc[1] < grid.n_cols):
            raise ValueError(f'{label} hors de la grille')
        if areas and any(g.covers(Point(port['lon'], float(lat_to_merc(port['lat'])))) for _, g in areas.surfaces):
            raise ValueError(f'{label} dans une zone F ; aucun deplacement automatique autorise')
        # Deplacement cotier original conserve, mais pas de fuite hors zone F.
        rc = snap_to_navigable(nav_cotes, rc)
        if masque_F[rc]:
            raise ValueError(f'Cellule cotiere du {label} interdite par une zone F')
        endpoints.append(rc)
    start, end = endpoints

    path_rc = grid_shortest_path(grid, nav, start, end, fully_connected=args.voisins == 8, speed_kn=speed if time_objective else None)
    lon_g, lat_g = grid.to_lonlat(path_rc[:, 0], path_rc[:, 1])
    d_grid = path_length_nm(lon_g, lat_g)
    print(f"[{time.time()-t0:5.1f}s] Dijkstra grille : {len(path_rc)} cellules, {d_grid:.1f} nm ({args.voisins} directions)", flush=True)

    def seg_len(p, q):
        lp, ap = grid.to_lonlat(p[0], p[1])
        lq, aq = grid.to_lonlat(q[0], q[1])
        return float(rhumb_nm(lp, ap, lq, aq))

    seg_cost = (lambda p,q: grid_segment_hours(grid, speed, p,q)) if time_objective else seg_len
    idx = string_pull(nav, path_rc, seg_cost if time_objective else None)
    lon_w, lat_w = grid.to_lonlat(path_rc[idx, 0], path_rc[idx, 1])
    print(f"[{time.time()-t0:5.1f}s] lissage glouton : {len(idx)} waypoints, {path_length_nm(lon_w, lat_w):.1f} nm", flush=True)

    idx = relax_waypoints(nav, path_rc, idx, seg_cost, respect_cost=time_objective)
    connectivity = args.voisins
    route_rc = path_rc[idx].astype(float)
    if (masque_F.any() or time_objective) and not all(line_of_sight(nav, p, q) for p, q in zip(route_rc[:-1], route_rc[1:])):
        print('Recalcul a quatre voisins : passage diagonal refuse par le lissage.', flush=True)
        path_rc = grid_shortest_path(grid, nav, start, end, fully_connected=False, speed_kn=speed if time_objective else None)
        lon_g, lat_g = grid.to_lonlat(path_rc[:, 0], path_rc[:, 1])
        d_grid = path_length_nm(lon_g, lat_g)
        idx = relax_waypoints(nav, path_rc, string_pull(nav, path_rc, seg_cost if time_objective else None), seg_cost, respect_cost=time_objective)
        route_rc = path_rc[idx].astype(float)
        connectivity = 4
    if (masque_F.any() or time_objective) and not all(line_of_sight(nav, p, q) for p, q in zip(route_rc[:-1], route_rc[1:])):
        raise RuntimeError('Trajet lisse non navigable')
    lon_w, lat_w = grid.to_lonlat(path_rc[idx, 0], path_rc[idx, 1])
    print(f"[{time.time()-t0:5.1f}s] optimisation locale : {len(idx)} waypoints, {path_length_nm(lon_w, lat_w):.1f} nm", flush=True)

    # Comportement original : remplacement par les positions exactes des ports.
    lon_w = np.concatenate([[departure["lon"]], lon_w[1:-1], [arrival["lon"]]])
    lat_w = np.concatenate([[departure["lat"]], lat_w[1:-1], [arrival["lat"]]])

    d_rhumb = path_length_nm(lon_w, lat_w, "rhumb")
    d_gc = path_length_nm(lon_w, lat_w, "gc")
    d_direct = float(haversine_nm(departure['lon'], departure['lat'], arrival['lon'], arrival['lat']))
    profile = route_speed_profile(lon_w, lat_w, speed_areas, args.vitesse)
    hours = sum(piece['duration_h'] for piece in profile)
    leg_hours = np.zeros(len(lon_w)-1)
    for piece in profile:
        leg_hours[piece['leg']] += piece['duration_h']
    cum_hours = np.r_[0., np.cumsum(leg_hours)]
    with (out_dir / 'profil_vitesse_W.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['leg','lon_start','lat_start','lon_end','lat_end','distance_nm','speed_kn','duration_h','zones_W'])
        writer.writeheader()
        for piece in profile:
            writer.writerow({**piece, 'zones_W': ' | '.join(piece['zones_W'])})
    np.savez_compressed(out_dir / 'vitesse_W.npz', vitesse_kn=speed,
                        lon0=grid.lon0,y_top=grid.y_top,res=grid.res,n_rows=grid.n_rows,n_cols=grid.n_cols)
    eta = departure['time'] + timedelta(hours=hours)
    planned_h = (arrival['time'] - departure['time']).total_seconds() / 3600

    check = check_route_against_land(polygons, lon_w, lat_w)
    violations_F = check_route_against_forbidden(areas, lon_w, lat_w) if areas else []
    if check['n_crossings'] or violations_F:
        raise RuntimeError(f'Trajet final invalide : {check["n_crossings"]} segment(s) sur terre ; zones F : {violations_F}')
    print(f"[{time.time()-t0:5.1f}s] verification exacte : {check['n_crossings']} segment(s) coupant la terre, "
          f"distance min a la cote ~{check['min_clearance_nm']:.2f} nm", flush=True)

    legs = rhumb_nm(lon_w[:-1], lat_w[:-1], lon_w[1:], lat_w[1:])
    cum = np.concatenate([[0.0], np.cumsum(legs)])
    with open(out_dir / f'waypoints_{route_tag}.csv', "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wp", "lat", "lon", "leg_nm", "cumul_nm", "eta_utc_vitesse_variable"])
        for k in range(len(lon_w)):
            t = departure["time"] + timedelta(hours=float(cum_hours[k]))
            w.writerow([k, f"{lat_w[k]:.5f}", f"{lon_w[k]:.5f}", f"{legs[k-1] if k else 0:.2f}",
                        f"{cum[k]:.2f}", t.strftime("%Y-%m-%d %H:%M")])

    geojson = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"from": departure['code'], "to": arrival['code'], "distance_nm": round(d_rhumb, 1),
                       "margin_nm": MARGIN_NM, "grid_res_deg": RES_DEG,
                       'zones_W_actives': not args.sans_W, 'duree_h': hours, 'vitesse_croisiere_kn': args.vitesse, 'unite_W_supposee': args.unite_W,
                       'zones_F_actives': areas is not None, 'raccord_hypothese': bool(areas and areas.iceberg_region_built)},
        "geometry": {"type": "LineString", "coordinates": [[float(a), float(b)] for a, b in zip(lon_w, lat_w)]},
    }]}
    (out_dir / f'route_{route_tag}.geojson').write_text(json.dumps(geojson, indent=1), encoding="utf-8")
    diagnostics = dict(distance_nm=d_rhumb, waypoints=len(lon_w),
        duree_h=hours, vitesse_croisiere_kn=args.vitesse, zones_W_actives=not args.sans_W,
        unite_W_supposee=args.unite_W, zones_W_lues=len(speed_areas), cellules_W_ralenties=slowed_cells,
        objectif='temps' if time_objective else 'distance_equivalente_temps_vitesse_constante',
        distance_en_W_nm=sum(p['distance_nm'] for p in profile if p['zones_W']),
        distance_ralentie_nm=sum(p['distance_nm'] for p in profile if p['speed_kn']<args.vitesse),
        zones_F_actives=areas is not None, cellules_F=int(masque_F.sum()),
        zones_F_lues=areas.source_F_count if areas else 0,
        raccord_hypothese=bool(areas and areas.iceberg_region_built),
        connexite_Dijkstra=connectivity, intersections_terre=check['n_crossings'],
        intersections_F=len(violations_F), bbox=list(args.bbox))
    (out_dir / 'diagnostic.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    if areas:
        np.savez_compressed(out_dir / 'masque_F.npz', masque_F=masque_F,
                            lon0=grid.lon0, y_top=grid.y_top, res=grid.res,
                            n_rows=grid.n_rows, n_cols=grid.n_cols)
        surfaces_json = dict(type='FeatureCollection',
            properties=dict(coordonnees='Mercator en degres equivalents, pas lon/lat',
                            raccord_hypothese=areas.iceberg_region_built),
            features=[dict(type='Feature', properties=dict(name=n), geometry=mapping(g)) for n, g in areas.surfaces])
        (out_dir / 'surfaces_F_mercator.json').write_text(json.dumps(surfaces_json), encoding='utf-8')

    summary = "\n".join([
        f"Route {departure['code']} -> {arrival['code']} : routage geographique",
        f" zones F : {areas is not None}, {int(masque_F.sum())} cellules ; Dijkstra {connectivity} voisins",
        f" grille Mercator {RES_DEG} deg, marge cote {MARGIN_NM} nm, {len(lon_w)} waypoints",
        f" distance (loxodromies entre waypoints) : {d_rhumb:8.1f} nm",
        f" distance (orthodromies entre waypoints): {d_gc:8.1f} nm",
        f" chemin Dijkstra brut (escalier) : {d_grid:8.1f} nm",
        f" vol d'oiseau (grand cercle direct) : {d_direct:8.1f} nm",
        (f" reference Port Distance Tool CMA CGM : {CMA_DISTANCE_NM:8.1f} nm (ecart {d_rhumb-CMA_DISTANCE_NM:+.1f} nm, {100*(d_rhumb/CMA_DISTANCE_NM-1):+.1f} %)" if not args.depart and not args.arrivee else ' points de demonstration : aucune reference commerciale applicable'),
        f" zones W : {not args.sans_W}, unite supposee {args.unite_W}, croisiere {args.vitesse} kn ; temps avec ralentissements : {hours:.2f} h",
        (f" depart {departure['time']:%d/%m/%Y %H:%M} UTC -> ETA {eta:%d/%m/%Y %H:%M} UTC "
        f"(arrivee prevue {arrival['time']:%d/%m/%Y %H:%M} UTC, soit {planned_h:.0f} h -> vitesse moyenne requise {d_rhumb/planned_h:.2f} kn)" if not args.depart and not args.arrivee else ' points de demonstration : pas de calendrier de voyage'),
        f" verification exacte vs polygones : {check['n_crossings']} segment coupant la terre, "
        f"distance min a la cote ~{check['min_clearance_nm']:.2f} nm",
    ])
    (out_dir / "resume.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary, flush=True)

    if not args.no_plot:
        plot_route(polygons, lon_w, lat_w, lon_g, lat_g, departure, arrival,
                   summary_title=f"{departure['code']} -> {arrival['code']} : {d_rhumb:.0f} nm, marge {MARGIN_NM} nm",
                   out_path=out_dir / f'carte_route_{route_tag}.png', extent=tuple(args.bbox), areas=areas, speed_areas=speed_areas)
    print(f"[{time.time()-t0:5.1f}s] termine -> {out_dir}", flush=True)
