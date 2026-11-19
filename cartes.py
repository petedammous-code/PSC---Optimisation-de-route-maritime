"""Representation des routes, des F et des W."""

import numpy as np
from shapely import box
from .configuration import BBOX
from .geometrie import lat_to_merc, merc_to_lat


SEA, LAND, COAST = "#dbe9f4", "#e9e4d8", "#9a927f"


ROUTE, GRID_PATH = "#d1495b", "#6c757d"


ZOOMS = [
    ("Detroit de Singapour", (103.4, 105.0, 0.8, 2.2)),
    ("Detroit de Taiwan", (117.5, 122.5, 21.5, 27.0)),
    ("Detroit de Bohai", (119.5, 123.0, 36.8, 40.2)),
]


def _draw(ax, polygons, lon_w, lat_w, lon_g, lat_g, extent, dep, arr, show_grid_path, areas=None, speed_areas=()):
    from matplotlib.collections import PolyCollection
    from matplotlib.ticker import FuncFormatter

    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_facecolor(SEA)
    verts = []
    for r in polygons:
        if r[:, 0].max() < lon_min - 1 or r[:, 0].min() > lon_max + 1 or r[:, 1].max() < lat_min - 1 or r[:, 1].min() > lat_max + 1:
            continue
        verts.append(np.column_stack([r[:, 0], lat_to_merc(r[:, 1])]))
    ax.add_collection(PolyCollection(verts, facecolor=LAND, edgecolor=LAND, linewidth=0.4))
    if areas or speed_areas:
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch
        from shapely.geometry.polygon import orient
        view = box(lon_min, float(lat_to_merc(lat_min)), lon_max, float(lat_to_merc(lat_max)))
        surfaces = [(g, '#bd3348') for _,g in areas.surfaces] if areas else []
        surfaces += [(z.geometry, '#d89b18') for z in speed_areas]
        for geometry, shade in surfaces:
            clipped = geometry.intersection(view)
            parts = [clipped] if clipped.geom_type == 'Polygon' else getattr(clipped, 'geoms', [])
            for part in parts:
                if part.is_empty or part.geom_type != 'Polygon':
                    continue
                part = orient(part, sign=1.0)
                paths = [MplPath(np.asarray(r.coords)) for r in [part.exterior, *part.interiors]]
                patch = PathPatch(MplPath.make_compound_path(*paths), facecolor=shade,
                                  edgecolor=shade, alpha=.25, linewidth=.6)
                ax.add_patch(patch)
    if show_grid_path:
        ax.plot(lon_g, lat_to_merc(lat_g), color=GRID_PATH, lw=0.8, ls=":", label="Dijkstra grille (escalier)")
    ax.plot(lon_w, lat_to_merc(lat_w), color=ROUTE, lw=1.6, label="Route lissee")
    ax.plot(lon_w[1:-1], lat_to_merc(lat_w[1:-1]), "o", color=ROUTE, ms=2.5)
    for p, c in ((dep, "#2a9d8f"), (arr, "#264653")):
        ax.plot(p["lon"], lat_to_merc(p["lat"]), "o", color=c, ms=7, mec="white", mew=1.2, zorder=5)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_to_merc(lat_min), lat_to_merc(lat_max))
    ax.set_aspect("equal")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(merc_to_lat(y)):.1f}°{'N' if merc_to_lat(y) >= 0 else 'S'}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.1f}°{'E' if x >= 0 else 'O'}"))
    ax.tick_params(labelsize=7)


def plot_route(polygons, lon_w, lat_w, lon_g, lat_g, dep, arr, summary_title, out_path, extent=None, areas=None, speed_areas=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if extent is not None and tuple(extent) != BBOX:
        fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
        _draw(ax, polygons, lon_w, lat_w, lon_g, lat_g, extent, dep, arr, False, areas, speed_areas)
        ax.set_title(summary_title + ('\nW en orange : zones traversables a vitesse limitee' if speed_areas else ''))
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(15, 11), dpi=150)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    _draw(ax, polygons, lon_w, lat_w, lon_g, lat_g, (99.5, 126.5, -1.5, 41.5), dep, arr, False, areas, speed_areas)
    ax.set_title(summary_title + ('\nW en orange : zones traversables a vitesse limitee' if speed_areas else ''), fontsize=12)
    ax.annotate(dep["code"], (dep["lon"], lat_to_merc(dep["lat"])), xytext=(6, -12), textcoords="offset points", fontsize=9)
    ax.annotate(arr["code"], (arr["lon"], lat_to_merc(arr["lat"])), xytext=(6, 4), textcoords="offset points", fontsize=9)
    for k, (name, ext) in enumerate(ZOOMS):
        a = fig.add_subplot(gs[k, 1])
        _draw(a, polygons, lon_w, lat_w, lon_g, lat_g, ext, dep, arr, True, areas, speed_areas)
        a.set_title(name, fontsize=10)
        lo0, lo1, la0, la1 = ext
        ax.add_patch(plt.Rectangle((lo0, lat_to_merc(la0)), lo1 - lo0, lat_to_merc(la1) - lat_to_merc(la0), fill=False, ec="#333", lw=0.8, ls="--"))
        if k == 0:
            a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
