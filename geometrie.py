"""Coordonnees Mercator et distances nautiques."""

import numpy as np


EARTH_RADIUS_NM = 6371008.8 / 1852.0


def haversine_nm(lon1, lat1, lon2, lat2):
    """Distance orthodromique en milles nautiques ; accepte des tableaux NumPy."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def rhumb_nm(lon1, lat1, lon2, lat2):
    """Distance loxodromique : longueur d'un segment droit dans le plan Mercator."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    dlam = (dlam + np.pi) % (2 * np.pi) - np.pi
    dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = np.where(np.abs(dpsi) > 1e-12, dphi / np.where(np.abs(dpsi) > 1e-12, dpsi, 1.0), np.cos(phi1))
    return EARTH_RADIUS_NM * np.sqrt(dphi ** 2 + (q * dlam) ** 2)


def lat_to_merc(lat):
    """Latitude en degres -> ordonnee Mercator en degres equivalents."""
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def merc_to_lat(y):
    """Ordonnee Mercator en degres equivalents -> latitude en degres."""
    return np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)


def path_length_nm(lons, lats, kind="rhumb"):
    """Longueur totale d'une route ; kind='rhumb' ou 'gc'."""
    lons, lats = np.asarray(lons), np.asarray(lats)
    f = rhumb_nm if kind == "rhumb" else haversine_nm
    return float(np.sum(f(lons[:-1], lats[:-1], lons[1:], lats[1:])))
