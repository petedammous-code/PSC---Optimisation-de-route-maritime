"""Construction et representation de la grille du routeur."""

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from .geometrie import lat_to_merc, merc_to_lat


@dataclass
class Grid:
    lon0: float                 # longitude du bord gauche
    y_top: float                # ordonnee Mercator du bord haut
    res: float                  # pas de grille en degres equivalents
    n_rows: int
    n_cols: int
    land: np.ndarray            # True = terre
    dist_land_nm: np.ndarray    # distance approx. a la terre, en milles nautiques
    cell_nm: np.ndarray         # taille au sol d'une cellule pour chaque ligne

    def to_rc(self, lon, lat):
        """Longitude/latitude -> ligne/colonne flottantes ; centres = entiers."""
        col = (np.asarray(lon) - self.lon0) / self.res - 0.5
        row = (self.y_top - lat_to_merc(np.asarray(lat))) / self.res - 0.5
        return row, col

    def to_lonlat(self, row, col):
        """Ligne/colonne -> longitude/latitude du centre de cellule."""
        lon = self.lon0 + (np.asarray(col) + 0.5) * self.res
        lat = merc_to_lat(self.y_top - (np.asarray(row) + 0.5) * self.res)
        return lon, lat

    def navigable(self, margin_nm: float) -> np.ndarray:
        return self.dist_land_nm > margin_nm


def build_grid(polygons: list[np.ndarray], bbox: tuple[float, float, float, float], res: float) -> Grid:
    """Rasterise les polygones de terre sur une grille Mercator."""
    lon_min, lon_max, lat_min, lat_max = bbox
    y_bottom, y_top = float(lat_to_merc(lat_min)), float(lat_to_merc(lat_max))
    n_cols = int(np.ceil((lon_max - lon_min) / res))
    n_rows = int(np.ceil((y_top - y_bottom) / res))

    img = Image.new("1", (n_cols, n_rows), 0)
    draw = ImageDraw.Draw(img)
    for ring in polygons:
        x = (ring[:, 0] - lon_min) / res - 0.5
        y = (y_top - lat_to_merc(ring[:, 1])) / res - 0.5
        pts = list(zip(x.tolist(), y.tolist()))
        if len(pts) >= 3:
            draw.polygon(pts, fill=1, outline=1)
    land = np.array(img, dtype=bool)

    lat_rows = merc_to_lat(y_top - (np.arange(n_rows) + 0.5) * res)
    cell_nm = res * 60.0 * np.cos(np.radians(lat_rows))
    # Correction de demi-diagonale conservee depuis le PDF.
    dist_cells = ndimage.distance_transform_edt(~land)
    dist_land_nm = np.maximum(dist_cells - np.sqrt(2) / 2, 0.0) * cell_nm[:, None]
    return Grid(lon_min, y_top, res, n_rows, n_cols, land, dist_land_nm, cell_nm)
