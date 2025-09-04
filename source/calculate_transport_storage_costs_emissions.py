"""
Date: August 22, 2025
Author: danikae
Purpose: Calculates costs and emissions associated with fuel transportation to the port and storage
"""

from load_inputs import load_global_parameters
from common_tools import get_top_dir, get_fuel_density
import os
import pandas as pd
import json
import math
import numpy as np
from typing import Optional, Tuple
import geopandas as gpd
from shapely.ops import transform as shp_transform
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer, Geod
import argparse
from typing import Iterable, Union
from shapely.geometry import LineString, MultiLineString
from land_transport_tools import calculate_land_transport_cost_emissions

KG_PER_TONNE = 1000
L_PER_CBM = 1000
KM_PER_NM = 1.852

PORT_COORD_OVERRIDES = {
    "port of singapore": (1.2640120293576107, 103.81925012028113),
    "kozmino": (42.72009799219331, 133.00860376409597),
    "rotterdam": (51.951479610560696, 4.145200170906766)
}

COUNTRY_ANCHORS = {
    "malaysia": (3.1477102684615104, 101.67484247279599),
}

glob = load_global_parameters()
top_dir = get_top_dir()

# Natural Earth admin_0 countries (prefer 10m, then 50m, then 110m)
NE10_URL  = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
NE50_URL  = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
NE110_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

NE10_LOCAL_ZIP  = os.path.join(top_dir, "cache", "natural_earth", "ne_10m_admin_0_countries.zip")
NE50_LOCAL_ZIP  = os.path.join(top_dir, "cache", "natural_earth", "ne_50m_admin_0_countries.zip")
NE110_LOCAL_ZIP = os.path.join(top_dir, "cache", "natural_earth", "ne_110m_admin_0_countries.zip")


# --- Country name normalization for Natural Earth matching
_COUNTRY_SYNONYMS = {
    "usa": "United States of America",
    "united states": "United States of America",
    "u.s.": "United States of America",
    "u.s.a.": "United States of America",
    "uk": "United Kingdom",
    "russia": "Russia",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "dr congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "congo (brazzaville)": "Congo",
    "uae": "United Arab Emirates",
    "czech republic": "Czechia",
    "burma": "Myanmar",
    "eswatini": "Eswatini",
    "bolivia": "Bolivia",
    "laos": "Laos",
    "syria": "Syria",
    "viet nam": "Vietnam",
}

#def calculate_land_transport_cost(fuel):
#    """
#    Calculates costs, in $/tonne to transport fuel by land to the port (currently assume pipeline)
#
#    Parameters
#    ----------
#    quantity : str
#        Quantity to make a bar for. Currently can be either cost or emissions
#
#    Returns
#    -------
#    cost_bar_dict : Dictionary
#        Dictionary containing data, colors, hatching, and labels for a cost bar
#    """
#    if "hydrogen" in fuel:
#        return glob["hydrogen_land_transport_cost"]["value"] * KG_PER_TONNE
#    if fuel == "ammonia":
#        return glob["ammonia_land_transport_cost"]["value"] * KG_PER_TONNE
#    if "ng" in fuel:
#        return glob["ng_land_transport_cost"]["value"] * glob["2016_to_2024_USD"]["value"] / glob["NG_density_STP"]["value"] * KG_PER_TONNE
#    if fuel == "methanol" or "diesel" in fuel or "bio" in fuel:
#        return glob["oil_land_transport_cost"]["value"] * glob["2016_to_2024_USD"]["value"] / (get_fuel_density(fuel) * L_PER_CBM) * KG_PER_TONNE
        
def get_countries():
    regional_tea_inputs_df = pd.read_csv(f"{top_dir}/input_fuel_pathway_data/regional_TEA_inputs.csv")
    countries = list(regional_tea_inputs_df["Region"])
    return countries
    
def load_departure_ports_table(filename: str = "singapore_rotterdam_departure_ports.csv") -> pd.DataFrame:
    """
    Reads the departure ports CSV you created earlier.
    Expected columns at least:
      - Region
      - Port of Singapore (string, never None for seaborne to Singapore)
      - Port of Rotterdam (string or None when land/pipeline dominates)
    """
    path = f"{top_dir}/input_fuel_pathway_data/transport/{filename}"
    df = pd.read_csv(path)
    df["Port of Rotterdam"] = df["Port of Rotterdam"].replace({"None": None, "none": None, "": None})
    df["Port of Singapore"] = df["Port of Singapore"].replace({"None": None, "none": None, "": None})
    return df
    
PORTS_REFERENCE_CSV = f"{top_dir}/input_fuel_pathway_data/transport/port_locations_ref.csv"


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two latitude/longitude points.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of the second point in decimal degrees.

    Returns
    -------
    float
        Great-circle distance in nautical miles (nm).
    """
    R_km = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return (R_km * c) * 0.539957  # km -> nm


def _try_load_ports_reference(filename: str) -> Optional[pd.DataFrame]:
    """
    Attempt to load a ports reference CSV file with columns for name, lat, lon (and optionally country).

    Parameters
    ----------
    filename : str
        Name of the ports reference CSV file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing at least columns ['name','lat','lon'] (and 'country' if available),
        or None if the file could not be loaded or does not contain the required columns.
    """
    path = f"{top_dir}/input_fuel_pathway_data/transport/{filename}"
    if not path or not os.path.exists(path):
        return None
    ref = pd.read_csv(path)
    cols = {c.lower(): c for c in ref.columns}

    def pick(*cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    name_col = pick("name", "port_name", "Main_Port_Name")
    lat_col  = pick("lat", "latitude", "Latitude")
    lon_col  = pick("lon", "longitude", "Longitude")
    country_col = pick("country", "Country")
    if not all([name_col, lat_col, lon_col]):
        return None
    out = ref.rename(columns={name_col: "name", lat_col: "lat", lon_col: "lon"})
    if country_col:
        out = out.rename(columns={country_col: "country"})
    return out[["name", "lat", "lon"] + (["country"] if "country" in out.columns else [])].dropna()


def _geocode_port_osm(name: str, country_hint: Optional[str]=None, pause: float=1.0) -> Optional[Tuple[float, float]]:
    """
    Use OpenStreetMap (via geopy) to geocode a port name into latitude and longitude.

    Parameters
    ----------
    name : str
        Name of the port.
    country_hint : str, optional
        Country name to improve geocoding accuracy.
    pause : float, optional
        Seconds to pause between queries to respect OSM rate limits (default = 1.0).

    Returns
    -------
    tuple of (float, float) or None
        Latitude and longitude in decimal degrees, or None if not found or geopy is unavailable.
    """
    try:
        from geopy.geocoders import Nominatim
    except Exception:
        return None
    geolocator = Nominatim(user_agent="fuel_routing_ports_locator")
    q = name if not country_hint else f"{name}, {country_hint}"
    try:
        loc = geolocator.geocode(q, timeout=15)
    except Exception:
        return None
    import time; time.sleep(pause)
    if loc:
        return float(loc.latitude), float(loc.longitude)
    return None


def _lookup_port_override(name: Optional[str]) -> Optional[Tuple[float,float]]:
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None
    key = str(name).strip().lower()
    return PORT_COORD_OVERRIDES.get(key)

# --- Update _resolve_port_latlon to check overrides FIRST
def _resolve_port_latlon(name: Optional[str],
                         ports_ref: Optional[pd.DataFrame],
                         country_hint: Optional[str]=None) -> Tuple[Optional[float], Optional[float]]:
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None, None

    # 1) hardcoded override
    ovr = _lookup_port_override(name)
    if ovr is not None:
        return ovr[0], ovr[1]  # (lat, lon)

    # 2) local reference CSV
    if ports_ref is not None and not ports_ref.empty:
        hit = ports_ref[ports_ref["name"].astype(str).str.lower() == str(name).lower()]
        if hit.empty:
            hit = ports_ref[ports_ref["name"].astype(str).str.lower().str.contains(str(name).lower())]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["lat"]), float(r["lon"])

    # 3) geocode fallback
    latlon = _geocode_port_osm(str(name), country_hint=country_hint)
    return (latlon[0], latlon[1]) if latlon else (None, None)

# Treat these as "empty"
_EMPTY_STRINGS = {"", "none", "nan", "null"}

def add_port_coordinates(
    df_ports: pd.DataFrame,
    ports_reference_csv: Optional[str] = PORTS_REFERENCE_CSV,
    verbose: bool = True,
    update_region_keyword: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Add latitude/longitude coordinates for Singapore and Rotterdam ports listed in df_ports.
    If a ports reference CSV already exists and contains coordinate columns, return it as-is
    unless update_region_keyword is provided — in which case, rows whose Region matches
    the keyword(s) will be (a) updated in-place and (b) created if they don't exist yet.
    """

    # --- helpers --------------------------------------------------------------
    def _normalize_kw(kws):
        if kws is None:
            return None
        if isinstance(kws, str):
            return {kws.strip().lower()}
        try:
            return {str(k).strip().lower() for k in kws if str(k).strip()}
        except TypeError:
            return {str(kws).strip().lower()}

    def _to_nan(x):
        if x is None:
            return np.nan
        if isinstance(x, float) and pd.isna(x):
            return np.nan
        if isinstance(x, str) and x.strip().lower() in _EMPTY_STRINGS:
            return np.nan
        return x

    def _safe_resolve_port_latlon(port_name, ref, country_hint=None) -> Tuple[Optional[float], Optional[float]]:
        if port_name is None:
            return (None, None)
        if isinstance(port_name, float) and pd.isna(port_name):
            return (None, None)
        if isinstance(port_name, str) and port_name.strip().lower() in _EMPTY_STRINGS:
            return (None, None)
        return _resolve_port_latlon(port_name, ref, country_hint=country_hint)

    def _try_make_parent_dir(path: Optional[str]):
        if path:
            outdir = os.path.dirname(path)
            if outdir:
                os.makedirs(outdir, exist_ok=True)

    # Normalize df_ports empties up-front
    for c in ["Port of Singapore", "Port of Rotterdam", "Port Country"]:
        if c in df_ports.columns:
            df_ports[c] = df_ports[c].map(_to_nan)

    kws = _normalize_kw(update_region_keyword)

    # --- if CSV exists with coordinate columns, maybe update in place ----------
    if ports_reference_csv and os.path.exists(ports_reference_csv):
        existing = pd.read_csv(ports_reference_csv)

        has_cols = all(
            c in existing.columns
            for c in ["pos_lat_sgp", "pos_lon_sgp", "pos_lat_rtm", "pos_lon_rtm"]
        )

        if has_cols and kws is None:
            if verbose:
                print(f"Using existing port coordinates from {ports_reference_csv}")
            return existing

        if has_cols and kws:
            if verbose:
                print(f"Updating/adding rows matching {sorted(kws)} in {ports_reference_csv}")

            needed_cols = [
                "Region",
                "Port of Singapore",
                "Port of Rotterdam",
                "Port Country",
            ]

            # Normalize existing empties as well
            for c in ["Port of Singapore", "Port of Rotterdam", "Port Country"]:
                if c in existing.columns:
                    existing[c] = existing[c].map(_to_nan)

            # Ensure required columns exist in 'existing'
            for c in needed_cols:
                if c not in existing.columns:
                    existing[c] = np.nan

            # Regions in existing matching the keywords
            region_series_existing = existing["Region"].astype(str)
            region_series_lower = region_series_existing.str.lower()
            mask_existing_matches = region_series_lower.apply(lambda x: any(kw in x for kw in kws))

            # Regions in df_ports matching the keywords (source of truth for new/updated rows)
            df_ports_regions = df_ports[needed_cols].drop_duplicates(subset=["Region"]).copy()
            df_ports_regions["__region_lower__"] = df_ports_regions["Region"].astype(str).str.lower()
            df_ports_matches = df_ports_regions[
                df_ports_regions["__region_lower__"].apply(lambda x: any(kw in x for kw in kws))
            ].drop(columns="__region_lower__", errors="ignore")

            # Identify which matching Regions are missing from 'existing'
            existing_regions_lower = set(region_series_lower.tolist())
            to_add = df_ports_matches[
                ~df_ports_matches["Region"].astype(str).str.lower().isin(existing_regions_lower)
            ].copy()

            # --- merge updated names into existing for rows that already exist ---
            if not df_ports_matches.empty:
                merged = existing.merge(
                    df_ports_matches,  # only matched regions
                    on="Region",
                    how="left",
                    suffixes=("", "_new"),
                )

                for c in ["Port of Singapore", "Port of Rotterdam", "Port Country"]:
                    newcol = f"{c}_new"
                    if newcol in merged.columns:
                        merged.loc[mask_existing_matches, c] = merged.loc[mask_existing_matches, newcol].map(_to_nan)

                drop_cols = [f"{c}_new" for c in needed_cols if f"{c}_new" in merged.columns]
                merged = merged.drop(columns=drop_cols)
                existing = merged

            # --- append new rows for Regions that didn't exist yet -------------
            if not to_add.empty:
                # Ensure coordinate columns exist before appending
                for c in ["pos_lat_sgp", "pos_lon_sgp", "pos_lat_rtm", "pos_lon_rtm"]:
                    if c not in existing.columns:
                        existing[c] = np.nan

                # Initialize coordinate columns for new rows
                for c in ["pos_lat_sgp", "pos_lon_sgp", "pos_lat_rtm", "pos_lon_rtm"]:
                    to_add[c] = np.nan

                # >>> FIX: align on union of columns before concat <<<
                union_cols = existing.columns.union(to_add.columns)
                existing_aligned = existing.reindex(columns=union_cols)
                to_add_aligned = to_add.reindex(columns=union_cols)
                existing = pd.concat([existing_aligned, to_add_aligned], ignore_index=True)

                if verbose:
                    added_names = ", ".join(to_add["Region"].astype(str).tolist())
                    print(f"Added new Region row(s): {added_names}")

            # --- resolve coords for all rows that match the keywords -----------
            ref = _try_load_ports_reference(ports_reference_csv)
            region_series_existing = existing["Region"].astype(str)
            region_series_lower = region_series_existing.str.lower()
            mask_all_matches = region_series_lower.apply(lambda x: any(kw in x for kw in kws))

            for idx in existing[mask_all_matches].index:
                row = existing.loc[idx]
                lat_s, lon_s = _safe_resolve_port_latlon(
                    row.get("Port of Singapore"),
                    ref,
                    country_hint=row.get("Port Country"),
                )
                lat_r, lon_r = _safe_resolve_port_latlon(
                    row.get("Port of Rotterdam"),
                    ref,
                    country_hint=row.get("Port Country"),
                )
                existing.at[idx, "pos_lat_sgp"] = lat_s
                existing.at[idx, "pos_lon_sgp"] = lon_s
                existing.at[idx, "pos_lat_rtm"] = lat_r
                existing.at[idx, "pos_lon_rtm"] = lon_r
                if verbose:
                    rgn = row.get("Region")
                    sgp_txt = f"({lat_s:.6f}, {lon_s:.6f})" if lat_s is not None and lon_s is not None else "CLEARED"
                    rtm_txt = f"({lat_r:.6f}, {lon_r:.6f})" if lat_r is not None and lon_r is not None else "CLEARED"
                    print(f"[update] {rgn}: SGP→{sgp_txt} RTM→{rtm_txt}")

            _try_make_parent_dir(ports_reference_csv)
            existing.to_csv(ports_reference_csv, index=False)
            if verbose:
                print(f"Saved updates to {ports_reference_csv}")
            return existing

        # CSV exists but lacks coordinate cols → fall through to compute all rows

    # --- (re)compute all rows --------------------------------------------------
    ref = _try_load_ports_reference(ports_reference_csv)
    df = df_ports.copy()
    n = len(df)

    def _resolve_and_log(port_name, country_hint, label, idx):
        lat, lon = _safe_resolve_port_latlon(port_name, ref, country_hint=country_hint)
        if verbose:
            if lat is not None and lon is not None:
                print(f"[{idx+1}/{n}] {label}: Resolved '{port_name}' → ({lat:.6f}, {lon:.6f})")
            else:
                shown = port_name if port_name is not None and not (isinstance(port_name, float) and pd.isna(port_name)) else "∅"
                print(f"[{idx+1}/{n}] {label}: No port (input={shown}) → CLEARED")
        return lat, lon

    pos_lat_sgp, pos_lon_sgp, pos_lat_rtm, pos_lon_rtm = [], [], [], []
    for idx, row in df.iterrows():
        lat_s, lon_s = _resolve_and_log(row.get("Port of Singapore"), row.get("Port Country"), "SGP", idx)
        lat_r, lon_r = _resolve_and_log(row.get("Port of Rotterdam"), row.get("Port Country"), "RTM", idx)
        pos_lat_sgp.append(lat_s); pos_lon_sgp.append(lon_s)
        pos_lat_rtm.append(lat_r); pos_lon_rtm.append(lon_r)

    df["pos_lat_sgp"] = pos_lat_sgp
    df["pos_lon_sgp"] = pos_lon_sgp
    df["pos_lat_rtm"] = pos_lat_rtm
    df["pos_lon_rtm"] = pos_lon_rtm

    if ports_reference_csv:
        _try_make_parent_dir(ports_reference_csv)
        df.to_csv(ports_reference_csv, index=False)
        if verbose:
            print(f"Saved resolved port coordinates to {ports_reference_csv}")

    return df



def _searoute_linestring(lon1: float, lat1: float, lon2: float, lat2: float):
    """
    Returns a shapely LineString (lon, lat) for the sea route between two points
    using searoute. Falls back to None if searoute or geometry is unavailable.
    """
    try:
        import searoute as sr
    except Exception:
        return None

    try:
        result = sr.searoute((float(lon1), float(lat1)), (float(lon2), float(lat2)))
        geom = result.get("geometry", {})
        coords = geom.get("coordinates")
        if not coords:
            return None
        ls = LineString(coords)
        # Some versions may return nested structures; normalize to LineString
        if isinstance(ls, (LineString, MultiLineString)):
            return ls
        return None
    except Exception:
        return None

def _searoute_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Uses the open-source 'searoute' package (OSM-based) to compute sailed distance in NM.
    Supports both the new properties format (length + units) and older keys.
    """
    try:
        import searoute as sr
    except Exception:
        return None

    try:
        # searoute takes (lon, lat)
        result = sr.searoute((float(lon1), float(lat1)), (float(lon2), float(lat2)))
        props = result.get("properties", {}) or {}

        # --- Preferred: new format 'length' + 'units'
        if "length" in props:
            val = float(props["length"])
            units = str(props.get("units", "")).lower()

            if units in {"nm", "nmi", "nautical_miles", "nautical-mile", "nautical mile"}:
                return val
            elif units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
                # 1 km = 1 / KM_PER_NM nautical miles
                return val / KM_PER_NM
            elif units in {"m", "meter", "meters", "metre", "metres"}:
                return val / 1852.0
            elif units in {"mi", "mile", "miles"}:
                # 1 statute mile = 1.609344 km
                return (val * 1.609344) / KM_PER_NM
            else:
                # If unknown, assume km as a reasonable default
                return val / KM_PER_NM

        # --- Back-compat: older keys
        if props.get("length_nm") is not None:
            return float(props["length_nm"])
        if props.get("length_m") is not None:
            return float(props["length_m"]) / 1852.0

        # --- Final fallback: sum haversine along geometry
        coords = result.get("geometry", {}).get("coordinates", [])
        if not coords or len(coords) < 2:
            return None
        total_nm = 0.0
        for (lon_a, lat_a), (lon_b, lat_b) in zip(coords[:-1], coords[1:]):
            total_nm += _haversine_nm(lat_a, lon_a, lat_b, lon_b)
        return total_nm
    except Exception:
        return None


def compute_sea_distances(
    df_with_coords: pd.DataFrame,
    prefer_searoute: bool = True,
    verbose: bool = True,
    log_every: int = 1
) -> pd.DataFrame:
    required = [
        "Region",
        "Port of Singapore","pos_lat_sgp","pos_lon_sgp",
        "Port of Rotterdam","pos_lat_rtm","pos_lon_rtm",
    ]
    missing = [c for c in required if c not in df_with_coords.columns]
    if missing:
        raise ValueError(f"compute_sea_distances: missing columns: {missing}")

    df = df_with_coords.copy()

    # Hubs (remember: cross-columns per your data layout)
    try:
        sgp_row = df.loc[df["Region"] == "Singapore"].iloc[0]
        lat_sgp, lon_sgp = float(sgp_row["pos_lat_rtm"]), float(sgp_row["pos_lon_rtm"])
    except Exception as e:
        raise ValueError("Need Singapore row with pos_lat_rtm/pos_lon_rtm populated.") from e

    try:
        nld_row = df.loc[df["Region"] == "Netherlands"].iloc[0]
        lat_rtm, lon_rtm = float(nld_row["pos_lat_sgp"]), float(nld_row["pos_lon_sgp"])
    except Exception as e:
        raise ValueError("Need Netherlands row with pos_lat_sgp/pos_lon_sgp populated.") from e

    sea_nm_to_sgp, sea_nm_to_rtm = [], []
    n = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        region = row["Region"]

        # --- Singapore leg: depart from row’s pos_*_sgp → (lat_sgp, lon_sgp)
        d_sgp, method_sgp = None, "none"
        lat_dep_sgp, lon_dep_sgp = row["pos_lat_sgp"], row["pos_lon_sgp"]
        if not any(pd.isna(x) for x in (lat_dep_sgp, lon_dep_sgp, lat_sgp, lon_sgp)):
            if prefer_searoute:
                d_try = _searoute_distance_nm(lat_dep_sgp, lon_dep_sgp, lat_sgp, lon_sgp)
                d_sgp, method_sgp = (d_try, "searoute") if d_try is not None else (_haversine_nm(lat_dep_sgp, lon_dep_sgp, lat_sgp, lon_sgp), "haversine")
            else:
                d_sgp, method_sgp = _haversine_nm(lat_dep_sgp, lon_dep_sgp, lat_sgp, lon_sgp), "haversine"
        sea_nm_to_sgp.append(d_sgp)

        # --- Rotterdam leg: 0 if land/pipeline; else row’s pos_*_rtm → (lat_rtm, lon_rtm)
        dep_port_rtm = row["Port of Rotterdam"]
        if pd.isna(dep_port_rtm) or (isinstance(dep_port_rtm, str) and dep_port_rtm.strip() == ""):
            d_rtm, method_rtm = 0.0, "land/pipeline"
        else:
            lat_dep_rtm, lon_dep_rtm = row["pos_lat_rtm"], row["pos_lon_rtm"]
            if any(pd.isna(x) for x in (lat_dep_rtm, lon_dep_rtm, lat_rtm, lon_rtm)):
                d_rtm, method_rtm = None, "none"
            else:
                if prefer_searoute:
                    d_try = _searoute_distance_nm(lat_dep_rtm, lon_dep_rtm, lat_rtm, lon_rtm)
                    d_rtm, method_rtm = (d_try, "searoute") if d_try is not None else (_haversine_nm(lat_dep_rtm, lon_dep_rtm, lat_rtm, lon_rtm), "haversine")
                else:
                    d_rtm, method_rtm = _haversine_nm(lat_dep_rtm, lon_dep_rtm, lat_rtm, lon_rtm), "haversine"
        sea_nm_to_rtm.append(d_rtm)

        if verbose and (idx % max(1, log_every) == 0 or idx == 1 or idx == n):
            parts = [f"[{idx}/{n}] {region}"]
            parts.append("SGP: missing coords" if d_sgp is None else f"SGP: {d_sgp:.0f} nm ({method_sgp})")
            if d_rtm is None:
                parts.append("RTM: missing coords")
            else:
                parts.append("RTM: 0 nm (land/pipeline)" if method_rtm == "land/pipeline" else f"RTM: {d_rtm:.0f} nm ({method_rtm})")
            print(" | ".join(parts))

    df["sea_nm_to_sgp"] = sea_nm_to_sgp
    df["sea_nm_to_rtm"] = sea_nm_to_rtm
    return df
    
def _normalize_country_name(name: str) -> str:
    k = str(name).strip()
    return _COUNTRY_SYNONYMS.get(k.lower(), k)

def _ensure_zip(url: str, local_zip: str) -> str:
    os.makedirs(os.path.dirname(local_zip), exist_ok=True)
    if not os.path.exists(local_zip):
        import urllib.request
        print(f"Downloading Natural Earth to {local_zip} ...")
        urllib.request.urlretrieve(url, local_zip)
    return local_zip

def _load_world() -> gpd.GeoDataFrame:
    """
    Load admin_0 countries in WGS84. Prefer 10m, then 50m, then 110m.
    Tries geodatasets if present; otherwise uses locally cached zips.
    """
    shp_path = None

    # 1) Try geodatasets (if installed) with several likely keys
    try:
        from geodatasets import get_path as gd_get_path
        candidates = [
            "naturalearth.cultural.admin_0_countries_10m",
            "naturalearth.admin_0_countries_10m",
            "naturalearth.cultural.admin_0_countries_50m",
            "naturalearth.admin_0_countries_50m",
            "naturalearth.cultural.admin_0_countries",
            "naturalearth.admin_0_countries",  # likely 110m
        ]
        for key in candidates:
            try:
                shp_path = gd_get_path(key)
                if shp_path:
                    break
            except Exception:
                pass
    except Exception:
        pass

    # 2) Fallback: cached zips (10m -> 50m -> 110m)
    if not shp_path:
        for url, local_zip in [
            (NE10_URL, NE10_LOCAL_ZIP),
            (NE50_URL, NE50_LOCAL_ZIP),
            (NE110_URL, NE110_LOCAL_ZIP),
        ]:
            try:
                shp_path = _ensure_zip(url, local_zip)
                break
            except Exception:
                continue
        if not shp_path:
            raise RuntimeError("Could not obtain any Natural Earth admin_0 dataset.")

    world = gpd.read_file(shp_path).to_crs("EPSG:4326")

    # Normalize a name column => 'name'
    name_col = None
    for cand in ["name", "NAME", "admin", "ADMIN", "sovereignt", "SOVEREIGNT"]:
        if cand in world.columns:
            name_col = cand
            break
    if name_col != "name":
        if name_col is None:
            raise ValueError(f"Country name column not found in dataset {shp_path}.")
        world = world.rename(columns={name_col: "name"})

    world = world[~world["name"].isna()].copy()
    world["name_norm"] = world["name"].astype(str).str.strip().str.lower()
    return world

def _get_country_geom(country: str, world_gdf: gpd.GeoDataFrame):
    """
    Return a (multi)polygon for the country if available; else a Point from geocoding.
    """
    if not country or pd.isna(country):
        return None
    cand = _normalize_country_name(country)
    sub = world_gdf.loc[world_gdf["name_norm"] == cand.lower()]
    if sub.empty:
        sub = world_gdf.loc[world_gdf["name_norm"].str.contains(cand.lower(), na=False)]
    if sub.empty:
        sub = world_gdf.loc[world_gdf["name_norm"].str.startswith(cand.lower(), na=False)]

    if not sub.empty:
        # GeoPandas ≥0.14 recommends union_all()
        try:
            return sub.geometry.union_all()
        except Exception:
            return sub.unary_union  # fallback for older versions

    # FINAL FALLBACK: geocode the country center as a point so distances still work
    latlon = _geocode_port_osm(cand, country_hint=None, pause=0.6)
    if latlon:
        return Point(latlon[1], latlon[0])  # (lon, lat)
    return None

def _mainland_only_geometry(geom, anchor_points=None):
    """
    Return only the 'mainland' (largest contiguous polygon) of a country's geometry.

    Rules:
      1) If anchor_points are provided, pick the polygon that contains the most anchors
         (ties → break on area).
      2) Otherwise, fall back to largest-by-area polygon (original behavior).

    If geom is already a Point or Polygon, return as-is.
    """

    if geom is None:
        return None
    if geom.geom_type in {"Point", "Polygon"}:
        return geom

    # Build local LAEA projection for area calculations
    rep = geom.representative_point()
    laea = CRS.from_proj4(f"+proj=laea +lat_0={rep.y} +lon_0={rep.x} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, laea, always_xy=True).transform
    inv = Transformer.from_crs(laea, wgs84, always_xy=True).transform

    geom_proj = shp_transform(fwd, geom)

    def _polygons(g):
        if g.geom_type == "Polygon":
            return [g]
        if g.geom_type == "MultiPolygon":
            return list(g.geoms)
        if g.geom_type == "GeometryCollection":
            polys = []
            for part in g.geoms:
                if part.geom_type == "Polygon":
                    polys.append(part)
                elif part.geom_type == "MultiPolygon":
                    polys.extend(list(part.geoms))
            return polys
        return []

    polys_proj = _polygons(geom_proj)
    if not polys_proj:
        return geom

    # If anchor points given, choose polygon containing the most anchors
    if anchor_points:
        polys_wgs84 = [shp_transform(inv, p) for p in polys_proj]
        counts = []
        for poly in polys_wgs84:
            c = sum(1 for pt in anchor_points if poly.contains(pt) or poly.touches(pt))
            counts.append((c, poly))
        max_count = max(c for c, _ in counts)
        candidates = [poly for c, poly in counts if c == max_count]
        if max_count > 0:
            if len(candidates) == 1:
                return candidates[0]
            # tie-break by area (using projected geometry)
            area_map = {id(poly): shp_transform(fwd, poly).area for poly in candidates}
            best = max(candidates, key=lambda poly: area_map[id(poly)])
            return best

    # Fallback: largest polygon by area
    best = max(polys_proj, key=lambda p: p.area)
    return shp_transform(inv, best)


def _centroid_geodesic(geom, country_name: str | None = None, mainland_only: bool = True):
    """
    Compute a robust centroid by centering a local Lambert Azimuthal Equal Area, taking
    the centroid there, then converting back to WGS84.

    If mainland_only is True, first reduce the geometry to its largest contiguous polygon,
    preferring the polygon that contains an anchor point if COUNTRY_ANCHORS has one for
    the provided country_name.
    """
    if geom is None:
        return None

    # Prepare optional anchor (if provided in COUNTRY_ANCHORS)
    anchor_points = None
    if mainland_only and country_name:
        key = country_name.strip().lower()
        if key in COUNTRY_ANCHORS:
            lat, lon = COUNTRY_ANCHORS[key]
            # Accept either (lat, lon) or (lon, lat); autodetect and normalize to (lon, lat)
            # Heuristic: if first looks like latitude and second like longitude, flip.
            if abs(lat) <= 90 and abs(lon) <= 180:
                anchor = Point(float(lon), float(lat))
            else:
                # If stored as (lon, lat) already
                anchor = Point(float(lat), float(lon))
            anchor_points = [anchor]

    if mainland_only:
        try:
            if anchor_points:
                geom = _mainland_only_geometry(geom, anchor_points=anchor_points)
            else:
                geom = _mainland_only_geometry(geom)
        except Exception:
            # safe fallback on any unexpected geometry issues
            pass

    # If the geometry is a point fallback, just return it
    if geom.geom_type == "Point":
        return geom

    # Project → centroid → back
    approx = geom.centroid
    laea = CRS.from_proj4(
        f"+proj=laea +lat_0={approx.y} +lon_0={approx.x} +datum=WGS84 +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, laea, always_xy=True).transform
    inv = Transformer.from_crs(laea, wgs84, always_xy=True).transform
    geom_proj = shp_transform(fwd, geom)
    cen_proj = geom_proj.centroid
    cen_wgs = shp_transform(inv, cen_proj)
    return cen_wgs  # shapely Point (lon, lat)


_GEOD_WGS84 = Geod(ellps="WGS84")

def _geodesic_km(lat1, lon1, lat2, lon2) -> float:
    # pyproj.Geod returns distance in meters
    _, _, dist_m = _GEOD_WGS84.inv(float(lon1), float(lat1), float(lon2), float(lat2))
    return dist_m / 1000.0

def _geodesic_linestring(lon1, lat1, lon2, lat2, npts: int = 128) -> LineString:
    """
    Approximate a geodesic with many small segments for plotting.
    """
    pts = _GEOD_WGS84.npts(lon1, lat1, lon2, lat2, npts)
    coords = [(lon1, lat1)] + pts + [(lon2, lat2)]
    return LineString(coords)

    
def enrich_with_country_centroids_and_distances(df_ports: pd.DataFrame,
                                                produce_maps: bool = False,
                                                maps_dir: Optional[str] = None,
                                                verbose: bool = True,
                                                countries_to_plot=None) -> pd.DataFrame:
    """
    Adds:
      - centroid_lat, centroid_lon
      - centroid_to_sgp_km, centroid_to_rtm_km

    Rules for distance columns:
      * If 'Port of Singapore' (or 'Port of Rotterdam') is None/blank → use geodesic
        distance from the country centroid directly to the destination port (land-only).
      * Otherwise → use geodesic distance from centroid to that departure port.

    Optionally writes per-country PNG maps showing centroid, legs, and (if applicable) sea routes.
    When a departure port is None/blank for a destination, the map draws only a solid
    centroid→destination geodesic (no sea segment, no departure port dot) for that destination.
    """
    # load Natural Earth once
    world = _load_world()

    # output columns
    cen_lat, cen_lon = [], []
    d_sgp_km, d_rtm_km = [], []

    # destination coordinates via overrides (lat, lon)
    lat_sgp_dest, lon_sgp_dest = PORT_COORD_OVERRIDES["port of singapore"]
    lat_rtm_dest, lon_rtm_dest = PORT_COORD_OVERRIDES["rotterdam"]

    # prepare map output
    if produce_maps:
        maps_dir = maps_dir or f"{top_dir}/input_fuel_pathway_data/transport/maps"
        os.makedirs(maps_dir, exist_ok=True)

    for i, row in df_ports.iterrows():
        country = row.get("Region")
        geom = _get_country_geom(country, world)
        if geom is None:
            if verbose:
                print(f"[centroid] Could not match country geometry for '{country}'.")
            cen_lat.append(np.nan); cen_lon.append(np.nan)
            d_sgp_km.append(np.nan); d_rtm_km.append(np.nan)
            continue

        # centroid (lon, lat), with mainland preference & anchors if available
        cen = _centroid_geodesic(geom, country, mainland_only=True)
        cen_lon.append(cen.x)
        cen_lat.append(cen.y)

        # helper: determine if a "port value" is effectively None/blank
        def _is_noneish(v) -> bool:
            return (v is None) or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip())

        # helper: choose target for centroid distance and flag if we go directly to destination
        def _centroid_to_target_km(port_value, dep_lat, dep_lon, dest_lat, dest_lon):
            if _is_noneish(port_value):
                # land path straight to destination
                return _geodesic_km(cen.y, cen.x, dest_lat, dest_lon), True
            # otherwise, to the departure port (if we have coords)
            if pd.isna(dep_lat) or pd.isna(dep_lon):
                return np.nan, False
            return _geodesic_km(cen.y, cen.x, dep_lat, dep_lon), False

        # compute distances + flags (used for mapping)
        ds, land_to_dest_sgp_flag = _centroid_to_target_km(
            row.get("Port of Singapore"),
            row.get("pos_lat_sgp"), row.get("pos_lon_sgp"),
            lat_sgp_dest, lon_sgp_dest
        )
        dr, land_to_dest_rtm_flag = _centroid_to_target_km(
            row.get("Port of Rotterdam"),
            row.get("pos_lat_rtm"), row.get("pos_lon_rtm"),
            lat_rtm_dest, lon_rtm_dest
        )
        d_sgp_km.append(ds)
        d_rtm_km.append(dr)

        # optional plotting per country
        if produce_maps and (countries_to_plot is None or country in countries_to_plot):
            try:
                # destinations as (lon, lat)
                dest_sgp = (lon_sgp_dest, lat_sgp_dest)
                dest_rtm = (lon_rtm_dest, lat_rtm_dest)

                # departure tuples if available; else None
                dep_sgp_tuple = None if land_to_dest_sgp_flag else (
                    (row.get("pos_lon_sgp"), row.get("pos_lat_sgp"))
                    if not any(pd.isna([row.get("pos_lat_sgp"), row.get("pos_lon_sgp")])) else None
                )
                dep_rtm_tuple = None if land_to_dest_rtm_flag else (
                    (row.get("pos_lon_rtm"), row.get("pos_lat_rtm"))
                    if not any(pd.isna([row.get("pos_lat_rtm"), row.get("pos_lon_rtm")])) else None
                )

                _plot_country_map(
                    country_name=str(country),
                    geom=geom,
                    centroid=(cen.x, cen.y),  # (lon, lat)
                    dep_sgp=dep_sgp_tuple,
                    dep_rtm=dep_rtm_tuple,
                    dest_sgp=dest_sgp,
                    dest_rtm=dest_rtm,
                    dep_name_sgp=(row.get("Port of Singapore") if row.get("Port of Singapore") else None),
                    dep_name_rtm=(row.get("Port of Rotterdam") if row.get("Port of Rotterdam") else None),
                    save_path=os.path.join(
                        maps_dir,
                        f"{_normalize_country_name(str(country)).replace(' ', '_')}_routes.png"
                    ),
                    land_to_dest_sgp=land_to_dest_sgp_flag,
                    land_to_dest_rtm=land_to_dest_rtm_flag,
                )

                if verbose:
                    print(f"[map] Saved {_normalize_country_name(str(country))} map")
            except Exception as e:
                if verbose:
                    print(f"[map] Failed for {country}: {e}")

    df_enriched = df_ports.copy()
    df_enriched["centroid_lat"] = cen_lat
    df_enriched["centroid_lon"] = cen_lon
    df_enriched["centroid_to_sgp_km"] = d_sgp_km
    df_enriched["centroid_to_rtm_km"] = d_rtm_km
    
    # Multiply the distance travelled over land by a factor of 1.3 based on typical detour factors of 1.2-1.4 used in literature (see eg. https://arxiv.org/pdf/2505.01124)
    detour_factor = 1.3
    df_enriched["centroid_to_sgp_km_pipeline"] = df_enriched["centroid_to_sgp_km"] * detour_factor
    df_enriched["centroid_to_rtm_km_pipeline"] = df_enriched["centroid_to_rtm_km"] * detour_factor
    return df_enriched

    
def _get_destination_coords(dest_name: str) -> Tuple[float, float]:
    """
    Returns (lat, lon) for a known destination port.
    Currently supports 'Singapore' and 'Rotterdam' via PORT_COORD_OVERRIDES.
    """
    key = dest_name.strip().lower()
    if key == "singapore":
        lat, lon = PORT_COORD_OVERRIDES["port of singapore"]
        return float(lat), float(lon)
    if key == "rotterdam":
        lat, lon = PORT_COORD_OVERRIDES["rotterdam"]
        return float(lat), float(lon)
    raise ValueError(f"Unknown destination {dest_name!r}")


def _plot_country_map(country_name: str,
                      geom,
                      centroid: Tuple[float, float],           # (lon, lat)
                      dep_sgp: Optional[Tuple[float, float]],   # (lon, lat) departure to Singapore
                      dep_rtm: Optional[Tuple[float, float]],   # (lon, lat) departure to Rotterdam (may be None)
                      dest_sgp: Tuple[float, float],            # (lon, lat) Port of Singapore
                      dest_rtm: Tuple[float, float],            # (lon, lat) Port of Rotterdam
                      dep_name_sgp: Optional[str],
                      dep_name_rtm: Optional[str],
                      save_path: str,
                      land_to_dest_sgp: bool = False,           # if True: draw centroid→destination (no sea, no dep dot)
                      land_to_dest_rtm: bool = False):          # if True: draw centroid→destination (no sea, no dep dot)
    """
    Basemap + mainland outline.

    Land legs:
      - If dep_* provided and land_to_dest_* is False: centroid → departure ports (red for SGP, blue for RTM).
      - If land_to_dest_* is True: centroid → destination port directly (solid).

    Sea legs:
      - Only when dep_* exists and land_to_dest_* is False: departure → destination (dashed).
      - Routes are segmented on anti-meridian jumps and wrapped to a stable longitude window.
      - Coincident sea routes are offset side-by-side after projection.

    Centroid: big star (no label). Departure ports: points (no labels, suppressed if land_to_dest_*).
    Destination ports: colored points + colored labels (SGP red, RTM blue).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import geopandas as gpd
    from shapely.geometry import Point, LineString, MultiLineString
    from shapely.ops import unary_union
    from shapely import affinity

    crs_wgs = "EPSG:4326"
    crs_web = "EPSG:3857"

    # ------------------ helpers: anti-meridian-safe routing ------------------
    def _segment_on_dateline(coords):
        """
        Split a path into segments whenever the longitude jump exceeds 180°.
        Returns a list of segments (each a list of (lon,lat)).
        """
        if not coords or len(coords) < 2:
            return [coords] if coords else []

        segs = []
        cur = [coords[0]]
        for i in range(1, len(coords)):
            lon0, lat0 = coords[i-1]
            lon1, lat1 = coords[i]
            if abs(lon1 - lon0) > 180.0:
                # close current segment, start a new one at i-1→i break
                if len(cur) >= 2:
                    segs.append(cur)
                cur = [coords[i]]  # start new segment at current point
            else:
                cur.append(coords[i])
        if len(cur) >= 2:
            segs.append(cur)
        return segs

    def _wrap_lon_window(lon, center):
        """Wrap longitude into (center-180, center+180] window."""
        x = lon
        while x <= center - 180.0:
            x += 360.0
        while x > center + 180.0:
            x -= 360.0
        return x

    def _wrap_segment_to_window(seg, center_lon):
        """Apply window wrap to all longitudes in a segment."""
        return [( _wrap_lon_window(lon, center_lon), lat ) for lon, lat in seg]

    def _sea_route_segments(a, b, center_lon):
        """
        Build anti-meridian-safe LineString segments for sea route a→b (lon,lat).
        - Uses your _searoute_linestring
        - Splits on large jumps
        - Wraps each segment to the plotting window
        """
        ls = _searoute_linestring(a[0], a[1], b[0], b[1])
        if ls is None or ls.is_empty:
            return []

        if isinstance(ls, (MultiLineString,)):
            # Merge to a single path when possible; if still multiple, pick longest piecewise but keep structure
            parts = []
            for geom in (ls.geoms if hasattr(ls, "geoms") else [ls]):
                coords = list(geom.coords)
                parts.extend(_segment_on_dateline(coords))
        else:
            coords = list(ls.coords)
            parts = _segment_on_dateline(coords)

        wrapped_parts = [_wrap_segment_to_window(seg, center_lon) for seg in parts]
        # convert to LineStrings (dropping any 1-point segments)
        return [LineString(seg) for seg in wrapped_parts if len(seg) >= 2]

    def _to_web_series(geoms):
        """Project list of WGS84 geoms to Web Mercator (handles empty)."""
        if not geoms:
            return gpd.GeoSeries([], crs=crs_wgs)
        return gpd.GeoSeries(geoms, crs=crs_wgs).to_crs(crs_web)

    # ------------------ mainland outline & plotting window center -----------
    is_poly = hasattr(geom, "geom_type") and geom.geom_type.lower() in {"polygon","multipolygon"}
    g_mainland_web = None
    mainland_center_lon = centroid[0]  # default
    if is_poly:
        try:
            mainland = _mainland_only_geometry(geom)
        except Exception:
            mainland = geom
        g_mainland = gpd.GeoSeries([mainland], crs=crs_wgs)
        g_mainland_web = g_mainland.to_crs(crs_web)
        # Use mainland bbox center (in WGS) for a stable wrapping window
        try:
            minx, miny, maxx, maxy = g_mainland.total_bounds
            mainland_center_lon = (minx + maxx) / 2.0
        except Exception:
            pass

    # ------------------ points (WGS → Web) ----------------------------------
    centroid_pt_wgs = Point(centroid[0], centroid[1])

    dep_pts_wgs = []
    if dep_sgp and not land_to_dest_sgp:
        dep_pts_wgs.append(Point(dep_sgp[0], dep_sgp[1]))
    if dep_rtm and not land_to_dest_rtm:
        dep_pts_wgs.append(Point(dep_rtm[0], dep_rtm[1]))
    g_dep_pts_web = _to_web_series(dep_pts_wgs)

    dest_pts_wgs = [Point(dest_sgp[0], dest_sgp[1]), Point(dest_rtm[0], dest_rtm[1])]
    g_dest_pts_web = gpd.GeoSeries(dest_pts_wgs, crs=crs_wgs).to_crs(crs_web)
    centroid_pt_web = gpd.GeoSeries([centroid_pt_wgs], crs=crs_wgs).to_crs(crs_web).iloc[0]

    # ------------------ land legs (geodesics) --------------------------------
    land_lines = []
    land_colors = []
    if dep_sgp or land_to_dest_sgp:
        to_lon, to_lat = (dest_sgp if land_to_dest_sgp else dep_sgp)
        land_lines.append(_geodesic_linestring(centroid[0], centroid[1], to_lon, to_lat))
        land_colors.append("red")
    if dep_rtm or land_to_dest_rtm:
        to_lon, to_lat = (dest_rtm if land_to_dest_rtm else dep_rtm)
        land_lines.append(_geodesic_linestring(centroid[0], centroid[1], to_lon, to_lat))
        land_colors.append("blue")

    g_land_web = _to_web_series(land_lines)

    # offset coincident land legs
    if len(g_land_web) == 2:
        try:
            if g_land_web.iloc[0].hausdorff_distance(g_land_web.iloc[1]) < 500:  # ~0.5 km
                off = 20_000
                try:
                    g_land_web.iloc[0] = g_land_web.iloc[0].parallel_offset(off, 'left', join_style=2)
                    g_land_web.iloc[1] = g_land_web.iloc[1].parallel_offset(off, 'right', join_style=2)
                except Exception:
                    g_land_web.iloc[0] = affinity.translate(g_land_web.iloc[0], xoff=off)
                    g_land_web.iloc[1] = affinity.translate(g_land_web.iloc[1], xoff=-off)
        except Exception:
            pass

    # ========================= SEA ROUTES (robust) ============================
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import unary_union

    def _densify_deg(coords, max_step_deg=0.5):
        """Densify in lon/lat so consecutive points are <= max_step_deg apart in either axis."""
        if not coords or len(coords) < 2:
            return coords
        out = [coords[0]]
        for (x0,y0),(x1,y1) in zip(coords[:-1], coords[1:]):
            dx, dy = x1 - x0, y1 - y0
            n = int(max(abs(dx), abs(dy)) // max_step_deg)
            if n > 0:
                for k in range(1, n+1):
                    t = k/(n+1)
                    out.append((x0 + dx*t, y0 + dy*t))
            out.append((x1,y1))
        return out

    def _unwrap_cumulative_lons(coords):
        """
        Cumulatively unwrap longitudes so successive steps are always the 'short way'
        around the globe, producing a monotonic-in-λ path (e.g., 170, 179, 181, 190, …).
        """
        if not coords:
            return coords
        unwrapped = [list(coords[0])]
        for (lon, lat) in coords[1:]:
            prev = unwrapped[-1][0]
            cur = lon
            # bring cur close to prev by adding/subtracting 360
            while (cur - prev) > 180.0:
                cur -= 360.0
            while (cur - prev) < -180.0:
                cur += 360.0
            unwrapped.append([cur, lat])
        return [tuple(p) for p in unwrapped]

    def _recentre_lons(coords, center_lon):
        """
        Shift all longitudes by ±360*k so that their mean is within (center-180, center+180].
        """
        if not coords:
            return coords
        lons = [c[0] for c in coords]
        mean_lon = sum(lons)/len(lons)
        # Move the whole sequence by multiples of 360 to be near center_lon
        shift = 0.0
        if mean_lon <= center_lon - 180.0:
            # shift right
            while (mean_lon + shift) <= center_lon - 180.0:
                shift += 360.0
        elif mean_lon > center_lon + 180.0:
            # shift left
            while (mean_lon + shift) > center_lon + 180.0:
                shift -= 360.0
        if shift:
            return [(lon + shift, lat) for lon, lat in coords]
        return coords

    def _route_polyline_wgs(a, b, center_lon):
        """
        Build a single anti-meridian-safe LineString for a→b:
          1) get searoute line,
          2) densify (stabilizes unwrap),
          3) cumulatively unwrap longitudes,
          4) recenter near the map window.
        """
        ls = _searoute_linestring(a[0], a[1], b[0], b[1])
        if not ls or ls.is_empty:
            return None
        if isinstance(ls, MultiLineString):
            # use the longest strand as the principal path
            ls = max(ls.geoms, key=lambda g: g.length)
        coords = list(ls.coords)
        coords = _densify_deg(coords, max_step_deg=0.75)
        coords = _unwrap_cumulative_lons(coords)
        coords = _recentre_lons(coords, center_lon)
        if len(coords) < 2:
            return None
        return LineString(coords)

    sea_lines_wgs = []
    sea_colors = []

    # Choose a stable center for recentering window; prefer mainland bbox in WGS, else centroid lon
    center_for_wrap = centroid[0]
    try:
        if is_poly:
            # compute from original WGS mainland, not projected
            g_mainland_wgs = gpd.GeoSeries([mainland], crs=crs_wgs)
            minx, miny, maxx, maxy = g_mainland_wgs.total_bounds
            center_for_wrap = (minx + maxx) / 2.0
    except Exception:
        pass

    if dep_sgp and not land_to_dest_sgp:
        ls = _route_polyline_wgs(dep_sgp, dest_sgp, center_for_wrap)
        if ls is not None:
            sea_lines_wgs.append(ls); sea_colors.append("red")

    if dep_rtm and not land_to_dest_rtm:
        ls = _route_polyline_wgs(dep_rtm, dest_rtm, center_for_wrap)
        if ls is not None:
            sea_lines_wgs.append(ls); sea_colors.append("blue")

    # project to Web Mercator
    g_sea_web = gpd.GeoSeries(sea_lines_wgs, crs=crs_wgs).to_crs(crs_web) if sea_lines_wgs else gpd.GeoSeries([], crs=crs_wgs)

    # --- FINAL anti-meridian guard: split projected lines on giant Δx jumps ---
    # World half-width in Web Mercator (meters)
    HALF_WORLD = 20037508.342789244
    # Treat anything larger than ~1/3 world width as a wrap seam
    DX_SEAM = HALF_WORLD / 3.0

    def _split_proj_on_large_dx(line):
        """Split a projected LineString into parts wherever Δx is huge (wrap seam)."""
        from shapely.geometry import LineString, MultiLineString
        if line.is_empty:
            return []
        coords = list(line.coords)
        if len(coords) < 2:
            return [line]

        parts = []
        cur = [coords[0]]
        for i in range(1, len(coords)):
            x0, y0 = coords[i-1]
            x1, y1 = coords[i]
            if abs(x1 - x0) > DX_SEAM:
                # close current segment if it has 2+ points
                if len(cur) >= 2:
                    parts.append(LineString(cur))
                cur = [coords[i]]  # start new segment
            else:
                cur.append(coords[i])
        if len(cur) >= 2:
            parts.append(LineString(cur))
        return parts

    if not g_sea_web.empty:
        new_geoms = []
        new_cols  = []
        for geom, col in zip(g_sea_web.geometry, sea_colors):
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                parts = _split_proj_on_large_dx(geom)
                new_geoms.extend(parts)
                new_cols.extend([col]*len(parts))
            elif geom.geom_type == "MultiLineString":
                for gg in geom.geoms:
                    parts = _split_proj_on_large_dx(gg)
                    new_geoms.extend(parts)
                    new_cols.extend([col]*len(parts))
            else:
                new_geoms.append(geom)
                new_cols.append(col)
        g_sea_web = gpd.GeoSeries(new_geoms, crs=crs_web)
        sea_colors = new_cols


    # ---------- robust overlap test + side-by-side offset ----------
    # If both exist and overlap substantially within a small buffer, offset them ±off_m.
    if len(g_sea_web) == 2:
        red_g = g_sea_web.iloc[0] if sea_colors[0] == "red" else g_sea_web.iloc[1]
        blue_g = g_sea_web.iloc[1] if sea_colors[0] == "red" else g_sea_web.iloc[0]

        try:
            tol_m = 1500.0
            red_buf = red_g.buffer(tol_m)
            blue_buf = blue_g.buffer(tol_m)
            overlap_area = red_buf.intersection(blue_buf).area
            min_area = min(red_buf.area, blue_buf.area)

            if min_area > 0 and (overlap_area / min_area) > 0.6:
                off_m = 20000.0

                def _offset(g, sign):
                    try:
                        # side-by-side; 'left' gives consistent lateral separation along the line
                        return g.parallel_offset(sign * off_m, 'left', join_style=2)
                    except Exception:
                        # fallback translate if offset fails
                        return affinity.translate(g, xoff=sign * off_m)

                # Apply offsets preserving original color order
                if sea_colors[0] == "red":
                    g_sea_web.iloc[0] = _offset(red_g, +1)
                    g_sea_web.iloc[1] = _offset(blue_g, -1)
                else:
                    g_sea_web.iloc[0] = _offset(blue_g, -1)
                    g_sea_web.iloc[1] = _offset(red_g, +1)
        except Exception:
            pass

    # Now g_sea_web + sea_colors are ready to plot (no dateline seam; twin routes offset).
    # ======================= END SEA ROUTES (robust) ==========================

    # ------------------ bounds / figure / draw -------------------------------
    bounds_sources = []
    if g_mainland_web is not None and not g_mainland_web.empty:
        bounds_sources.append(g_mainland_web)
    if not g_land_web.empty:
        bounds_sources.append(g_land_web)
    if not g_sea_web.empty:
        bounds_sources.append(g_sea_web)
    if not g_dep_pts_web.empty:
        bounds_sources.append(g_dep_pts_web)
    bounds_sources.append(g_dest_pts_web)

    if bounds_sources:
        xmin, ymin, xmax, ymax = unary_union([s.unary_union for s in bounds_sources]).bounds
    else:
        xmin = ymin = -1e4; xmax = ymax = 1e4

    # padding + minimum span
    dx, dy = xmax - xmin, ymax - ymin
    pad = 0.10
    xmin -= dx*pad; xmax += dx*pad; ymin -= dy*pad; ymax += dy*pad
    min_span_m = 250_000
    if (xmax - xmin) < min_span_m:
        midx = (xmin + xmax)/2; xmin, xmax = midx - min_span_m/2, midx + min_span_m/2
    if (ymax - ymin) < min_span_m:
        midy = (ymin + ymax)/2; ymin, ymax = midy - min_span_m/2, midy + min_span_m/2

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    tiles_ok, fail_msg = False, ""
    try:
        import contextily as ctx
        for src in (ctx.providers.CartoDB.Positron, ctx.providers.OpenStreetMap.Mapnik):
            try:
                img, ext = ctx.bounds2img(xmin, ymin, xmax, ymax, source=src, ll=False)
                ax.imshow(img, extent=ext, interpolation="bilinear", zorder=1)
                tiles_ok = True
                break
            except Exception as e:
                fail_msg = str(e)
    except Exception as e:
        fail_msg = f"contextily import failed: {e}"
    if not tiles_ok:
        ax.text((xmin+xmax)/2, (ymin+ymax)/2,
                "Basemap unavailable\n" + fail_msg, ha="center", va="center",
                fontsize=10, alpha=0.8)

    # mainland outline
    if g_mainland_web is not None:
        g_mainland_web.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0, zorder=10)

    # sea legs (dashed, color)
    if not g_sea_web.empty:
        for geom, col in zip(g_sea_web, sea_colors):
            gpd.GeoSeries([geom], crs=crs_web).plot(
                ax=ax, linewidth=1.5, alpha=0.95, zorder=14, color=col, linestyle="--"
            )

    # land legs (solid, color)
    if not g_land_web.empty:
        for geom, col in zip(g_land_web, land_colors):
            gpd.GeoSeries([geom], crs=crs_web).plot(
                ax=ax, linewidth=1.5, alpha=0.95, zorder=15, color=col,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
            )

    # centroid star
    ax.scatter([centroid_pt_web.x], [centroid_pt_web.y], s=100, marker='*', zorder=20)

    # departure dots (no labels)
    if not g_dep_pts_web.empty:
        g_dep_pts_web.plot(ax=ax, markersize=40, color="#333", edgecolor="white", linewidth=1.0, zorder=18)

    # destination dots + labels (SGP red, RTM blue; RTM label to the left, slightly up; SGP to the right)
    dest_labels = ["Singapore", "Rotterdam"]
    dest_colors = ["red", "blue"]

    for pt, lab, col in zip(g_dest_pts_web.geometry, dest_labels, dest_colors):
        ax.scatter(pt.x, pt.y, s=60, color=col, edgecolor="white", linewidth=1.2, zorder=22)
        if lab == "Rotterdam":
            xytext = (-3, 3); ha = "right"; va = "bottom"
        else:
            xytext = (4, 4); ha = "left"; va = "bottom"
        ax.annotate(
            lab, (pt.x, pt.y), xytext=xytext, textcoords="offset points",
            fontsize=10, zorder=23, ha=ha, va=va, color=col,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", action="store_true",
                        help="Produce per-country PNG maps with centroid, ports, and geodesic lines.")
    args = parser.parse_args()
 
    fuels = ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "methanol", "FTdiesel", "lng", "bio_cfp", "bio_leo"]
    countries = get_countries()
    destination_ports = ["Singapore", "Rotterdam"]
    
    departure_ports_df = load_departure_ports_table()
    departure_ports_df = add_port_coordinates(departure_ports_df, update_region_keyword="philippines")#"malaysia"
    departure_ports_df = compute_sea_distances(departure_ports_df)
    
    departure_ports_df = enrich_with_country_centroids_and_distances(
        departure_ports_df,
        produce_maps=bool(args.maps),
        maps_dir=f"{top_dir}/input_fuel_pathway_data/transport/maps",
        verbose=True,
        countries_to_plot=None #["United States"]
    )
    
    departure_ports_df.to_csv(f"{top_dir}/input_fuel_pathway_data/transport/singapore_rotterdam_departure_ports_with_distances.csv")
    departure_ports_df.set_index("Region", inplace=True)
    
    for fuel in fuels:
        
        for dest in destination_ports:
            rows = []
            for country in countries:
                
                # Calculate fuel transport cost over land
                if dest == "Singapore":
                    land_transport_km = departure_ports_df.loc[country, "centroid_to_sgp_km_pipeline"]   # Distance travelled over land, in km
                elif dest == "Rotterdam":
                    land_transport_km = departure_ports_df.loc[country, "centroid_to_rtm_km_pipeline"]
                else:
                    print(f"Error: Destination {dest} not yet supported. Currently support Singapore and Rotterdam as destination ports.")
                
                land_transport_cost, land_transport_emissions = calculate_land_transport_cost_emissions(fuel, land_transport_km)
                
                row = {
                    "Region": country,
                    "Fuel": fuel,
                    "Land Transport Cost [$/tonne]": land_transport_cost,
                    "Land Transport Emissions [kg CO2e / kg fuel]": land_transport_emissions,
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            # Stable column order; future columns will be appended automatically
            base_cols = ["Region", "Fuel", "Land Transport Cost [$/tonne]", "Land Transport Emissions [kg CO2e / kg fuel]"]
            extra_cols = [c for c in df.columns if c not in base_cols]
            df = df[base_cols + extra_cols]

            os.makedirs(f"{top_dir}/input_fuel_pathway_data/transport", exist_ok=True)
            df.to_csv(f"{top_dir}/input_fuel_pathway_data/transport/{fuel}_{dest}.csv", index=False)
            print(f"Saved transport costs and emissions to {top_dir}/input_fuel_pathway_data/transport/{fuel}_{dest}.csv")

if __name__ == "__main__":
    main()
