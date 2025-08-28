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

KG_PER_TONNE = 1000
L_PER_CBM = 1000
KM_PER_NM = 1.852

glob = load_global_parameters()
top_dir = get_top_dir()

def calculate_land_transport_cost(fuel):
    """
    Calculates costs, in $/tonne to transport fuel by land to the port (currently assume pipeline)

    Parameters
    ----------
    quantity : str
        Quantity to make a bar for. Currently can be either cost or emissions

    Returns
    -------
    cost_bar_dict : Dictionary
        Dictionary containing data, colors, hatching, and labels for a cost bar
    """
    if "hydrogen" in fuel:
        return glob["hydrogen_land_transport_cost"]["value"] * KG_PER_TONNE
    if fuel == "ammonia":
        return glob["ammonia_land_transport_cost"]["value"] * KG_PER_TONNE
    if "ng" in fuel:
        return glob["ng_land_transport_cost"]["value"] * glob["2016_to_2024_USD"]["value"] / glob["NG_density_STP"]["value"] * KG_PER_TONNE
    if fuel == "methanol" or "diesel" in fuel or "bio" in fuel:
        return glob["oil_land_transport_cost"]["value"] * glob["2016_to_2024_USD"]["value"] / (get_fuel_density(fuel) * L_PER_CBM) * KG_PER_TONNE
        
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
    path = f"{top_dir}/input_fuel_pathway_data/{filename}"
    df = pd.read_csv(path)
    df["Port of Rotterdam"] = df["Port of Rotterdam"].replace({"None": None, "none": None, "": None})
    df["Port of Singapore"] = df["Port of Singapore"].replace({"None": None, "none": None, "": None})
    return df
    
PORTS_REFERENCE_CSV = f"{top_dir}/input_fuel_pathway_data/port_locations_ref.csv"


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
    path = f"{top_dir}/input_fuel_pathway_data/{filename}"
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


def _resolve_port_latlon(name: Optional[str],
                         ports_ref: Optional[pd.DataFrame],
                         country_hint: Optional[str]=None) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve the latitude and longitude of a port, using a local reference table if available,
    and falling back to OSM geocoding otherwise.

    Parameters
    ----------
    name : str or None
        Name of the port.
    ports_ref : pandas.DataFrame or None
        Ports reference DataFrame with columns 'name','lat','lon' (optional 'country').
    country_hint : str, optional
        Country name to help with geocoding.

    Returns
    -------
    (lat, lon) : tuple of floats or (None, None)
        Coordinates of the port in decimal degrees, or (None, None) if not found.
    """
    if not name or (isinstance(name, float) and pd.isna(name)):
        return None, None
    if ports_ref is not None and not ports_ref.empty:
        hit = ports_ref[ports_ref["name"].astype(str).str.lower() == str(name).lower()]
        if hit.empty:
            hit = ports_ref[ports_ref["name"].astype(str).str.lower().str.contains(str(name).lower())]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["lat"]), float(r["lon"])
    latlon = _geocode_port_osm(str(name), country_hint=country_hint)
    return (latlon[0], latlon[1]) if latlon else (None, None)


def add_port_coordinates(df_ports: pd.DataFrame,
                         ports_reference_csv: Optional[str] = PORTS_REFERENCE_CSV,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Add latitude/longitude coordinates for Singapore and Rotterdam ports
    listed in the provided DataFrame.

    If a ports reference CSV already exists and contains coordinate columns,
    those are returned directly and no new resolution is attempted.

    Parameters
    ----------
    df_ports : pandas.DataFrame
        DataFrame with columns 'Port of Singapore', 'Port of Rotterdam', and 'Port Country'.
    ports_reference_csv : str, optional
        Path to ports reference CSV for faster lookups (default = PORTS_REFERENCE_CSV env variable).
    verbose : bool, optional
        Print progress as ports are resolved (default = True).

    Returns
    -------
    pandas.DataFrame
        Copy of df_ports with new columns:
        - pos_lat_sgp, pos_lon_sgp : coordinates of Port of Singapore
        - pos_lat_rtm, pos_lon_rtm : coordinates of Port of Rotterdam
    """
    # If CSV exists and already has coordinates, just load and return it
    if ports_reference_csv and os.path.exists(ports_reference_csv):
        existing = pd.read_csv(ports_reference_csv)
        if all(c in existing.columns for c in ["pos_lat_sgp", "pos_lon_sgp", "pos_lat_rtm", "pos_lon_rtm"]):
            if verbose:
                print(f"Using existing port coordinates from {ports_reference_csv}")
            return existing

    # Otherwise resolve coordinates as before
    ref = _try_load_ports_reference(ports_reference_csv)
    df = df_ports.copy()
    n = len(df)

    def _resolve_and_log(port_name, country_hint, label, idx):
        lat, lon = _resolve_port_latlon(port_name, ref, country_hint=country_hint)
        if verbose:
            if lat is not None and lon is not None:
                print(f"[{idx+1}/{n}] {label}: Resolved '{port_name}' → ({lat:.4f}, {lon:.4f})")
            else:
                print(f"[{idx+1}/{n}] {label}: Could not resolve '{port_name}'")
        return lat, lon

    pos_lat_sgp, pos_lon_sgp, pos_lat_rtm, pos_lon_rtm = [], [], [], []
    for idx, row in df.iterrows():
        lat_s, lon_s = _resolve_and_log(row.get("Port of Singapore"),
                                        row.get("Port Country"), "SGP", idx)
        lat_r, lon_r = _resolve_and_log(row.get("Port of Rotterdam"),
                                        row.get("Port Country"), "RTM", idx)
        pos_lat_sgp.append(lat_s)
        pos_lon_sgp.append(lon_s)
        pos_lat_rtm.append(lat_r)
        pos_lon_rtm.append(lon_r)

    df["pos_lat_sgp"] = pos_lat_sgp
    df["pos_lon_sgp"] = pos_lon_sgp
    df["pos_lat_rtm"] = pos_lat_rtm
    df["pos_lon_rtm"] = pos_lon_rtm

    # Save to CSV so next time it can be reused
    if ports_reference_csv:
        os.makedirs(os.path.dirname(ports_reference_csv), exist_ok=True)
        df.to_csv(ports_reference_csv, index=False)
        if verbose:
            print(f"Saved resolved port coordinates to {ports_reference_csv}")

    return df

def _searoute_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Uses the open-source 'searoute' package (OSM-based) to compute sailed NM.
    Returns None if searoute isn't installed or routing fails.
    """
    try:
        import searoute as sr
    except Exception:
        return None

    try:
        # searoute takes (lon, lat)
        result = sr.searoute((float(lon1), float(lat1)), (float(lon2), float(lat2)))
        # Many builds expose distance as either 'length_nm' or 'length_m'
        props = result.get("properties", {})
        if "length_nm" in props and props["length_nm"] is not None:
            return float(props["length_nm"])
        if "length_m" in props and props["length_m"] is not None:
            return float(props["length_m"]) / 1852.0
        # Fallback: sum segment haversines along the returned line geometry
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


def main():
 
    fuels = ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "methanol", "FTdiesel", "lng", "bio_cfp", "bio_leo"]
    countries = get_countries()
    destination_ports = ["Singapore", "Rotterdam"]
    
    departure_ports_df = load_departure_ports_table()
    departure_ports_df = add_port_coordinates(departure_ports_df)
    departure_ports_df = compute_sea_distances(departure_ports_df)
    departure_ports_df.to_csv(f"{top_dir}/input_fuel_pathway_data/singapore_rotterdam_departure_ports_with_distances.csv")
    
    """
    for fuel in fuels:
        land_transport_cost = calculate_land_transport_cost(fuel)
        
        for dest in destination_ports:
            rows = []
            for country in countries:
                row = {
                    "Region": country,
                    "Fuel": fuel,
                    "Land Transport Cost (2024$/tonne)": land_transport_cost,
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            # Stable column order; future columns will be appended automatically
            base_cols = ["Region", "Fuel", "Land Transport Cost (2024$/tonne)"]
            extra_cols = [c for c in df.columns if c not in base_cols]
            df = df[base_cols + extra_cols]

            os.makedirs(f"{top_dir}/input_fuel_pathway_data/transport", exist_ok=True)
            df.to_csv(f"{top_dir}/input_fuel_pathway_data/transport/{fuel}_{dest}.csv", index=False)
            print(f"Saved transport costs and emissions to {top_dir}/input_fuel_pathway_data/transport/{fuel}_{dest}.csv")
    """
if __name__ == "__main__":
    main()
