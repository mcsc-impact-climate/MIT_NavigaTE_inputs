import os
from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd


def get_top_dir():
    """
    Gets the path to the top level of the git repo (one level up from the source directory)

    Parameters
    ----------
    None

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
    """
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    top_dir = os.path.dirname(source_dir)
    return top_dir


def ensure_directory_exists(directory_path):
    """
    Checks if the specified directory exists, and if not, creates it.

    Parameters
    ----------
    directory_path (str): The path of the directory to check or create.

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def generate_blue_shades(num_shades):
    """
    Generates a list of blue shades ranging from light to dark.

    Parameters
    ----------
    num_shades : int
        The number of blue shades to generate.

    Returns
    -------
    blue_shades : list of str
        A list of blue shades in hex format, ranging from light to dark.
    """
    # Define the start and end colors (light blue to dark blue)
    light_blue = mcolors.to_rgba("#add8e6")  # Light blue
    dark_blue = mcolors.to_rgba("#00008b")  # Dark blue

    # Create a list of colors by interpolating between light blue and dark blue
    if num_shades > 1:
        blue_shades = [
            mcolors.to_hex(
                (
                    dark_blue[0] * (1 - i / (num_shades - 1))
                    + light_blue[0] * (i / (num_shades - 1)),
                    dark_blue[1] * (1 - i / (num_shades - 1))
                    + light_blue[1] * (i / (num_shades - 1)),
                    dark_blue[2] * (1 - i / (num_shades - 1))
                    + light_blue[2] * (i / (num_shades - 1)),
                    1.0,
                )
            )
            for i in range(num_shades)
        ]
    else:
        blue_shades = [light_blue]

    return blue_shades


def get_pathway_type(pathway, info_file=None):
    """
    Reads in the pathway type label for the given fuel production pathway based on the csv info file.

    Parameters
    ----------
    pathway : str
        Name of the pathway

    info_file : str
        Path to an info file that contains a mapping between pathway names and types

    Returns
    -------
    pathway_type : str
        Pathway type associated with the given pathway name
    """
    if info_file is None:
        top_dir = get_top_dir()
        info_file = f"{top_dir}/info_files/pathway_info.csv"

    try:
        info_df = pd.read_csv(info_file)
    except FileNotFoundError:
        raise Exception(
            f"Pathway info file {info_file} not found. Cannot evaluate pathway type."
        )
    try:
        pathway_type = info_df["Pathway Type"][info_df["Pathway Name"] == pathway].iloc[
            0
        ]
    except KeyError as e:
        raise Exception(
            f"KeyError: {e.args[0]} not found in the provided info file {info_file}. Cannot evaluate pathway type."
        )
    return pathway_type


def get_pathway_type_color(pathway_type, info_file=None):
    if info_file is None:
        top_dir = get_top_dir()
        info_file = f"{top_dir}/info_files/pathway_type_info.csv"

    try:
        info_df = pd.read_csv(info_file)
    except FileNotFoundError:
        raise Exception(
            f"Pathway info file {info_file} not found. Cannot evaluate pathway color."
        )
    try:
        pathway_color = info_df["Color"][info_df["Pathway Type"] == pathway_type].iloc[
            0
        ]
    except KeyError as e:
        raise Exception(
            f"KeyError: {e.args[0]} not found in the provided info file {info_file}. Cannot evaluate pathway color."
        )
    return pathway_color


def get_pathway_type_label(pathway_type, info_file=None):
    if info_file is None:
        top_dir = get_top_dir()
        info_file = f"{top_dir}/info_files/pathway_type_info.csv"
    try:
        info_df = pd.read_csv(info_file)
    except FileNotFoundError:
        raise Exception(
            f"Pathway info file {info_file} not found. Cannot evaluate pathway type label."
        )
    try:
        pathway_type_label = info_df["Label"][
            info_df["Pathway Type"] == pathway_type
        ].iloc[0]
    except KeyError as e:
        raise Exception(
            f"KeyError: {e.args[0]} not found in the provided info file {info_file}. Cannot evaluate pathway type label."
        )
    return pathway_type_label


def read_pathway_labels():
    """
    Reads in the label and description of each fuel production pathway from an info file.

    Parameters
    ----------
    top_dir: str
        Absolute path to the top level of the repo

    Returns
    -------
    pathway_labels_df : pandas.DataFrame
        Dataframe containing the label and description of each fuel production pathway.
    """
    top_dir = get_top_dir()
    pathway_labels_df = pd.read_csv(f"{top_dir}/info_files/pathway_info.csv").set_index(
        "Pathway Name"
    )
    return pathway_labels_df


def get_pathway_label(pathway):
    """
    Returns the label in the row of the pathway_labels_df corresponding to the given pathway.

    Parameters
    ----------
    pathway_labels_df: pandas DataFrame
        Dataframe containing labels and descriptions for each fuel production pathway

    pathway : str
        String identifier for the pathway of interest

    Returns
    -------
    pathway_label : str
        User-friendly label corresponding to the given pathway ID.
    """
    pathway_labels_df = read_pathway_labels()
    return pathway_labels_df.loc[pathway, "Label"]


def read_fuel_labels():
    """
    Reads in the label and description of each fuel from an info file.

    Parameters
    ----------
    top_dir: str
        Absolute path to the top level of the repo

    Returns
    -------
    fuel_labels_df : pandas.DataFrame
        Dataframe containing the label and description of each fuel.
    """
    top_dir = get_top_dir()
    fuel_labels_df = pd.read_csv(f"{top_dir}/info_files/fuel_info.csv").set_index(
        "Fuel"
    )
    return fuel_labels_df


def get_fuel_label(fuel):
    """
    Returns the label in the row of the fuel_labels_df corresponding to the given fuel.

    Parameters
    ----------
    fuel : str
        String identifier for the fuel of interest

    Returns
    -------
    fuel_label : str
        User-friendly label corresponding to the given fuel
    """
    fuel_labels_df = read_fuel_labels()
    return fuel_labels_df.loc[fuel, "Label"]


def get_fuel_LHV(fuel):
    """
    Returns the LHV of the given fuel.

    Parameters
    ----------
    fuel : str
        String identifier for the fuel of interest

    Returns
    -------
    fuel_LHV : str
        LHV the given fuel
    """
    fuel_info_df = read_fuel_labels()
    return fuel_info_df.loc[fuel, "Lower Heating Value (MJ / kg)"]


def get_fuel_density(fuel):
    """
    Returns the density of the given fuel.

    Parameters
    ----------
    fuel : str
        String identifier for the fuel of interest

    Returns
    -------
    fuel_density : str
        Density the given fuel
    """
    fuel_info_df = read_fuel_labels()
    return fuel_info_df.loc[fuel, "Mass density (kg/L)"]


def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it doesn't already exist.

    Parameters:
    ----------
    directory_path : str
        The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
