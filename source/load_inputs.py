import json
import os

from common_tools import get_top_dir

top_dir = get_top_dir()


def load_molecular_info(filepath=None):
    if filepath is None:
        filepath = os.path.join(
            top_dir, "input_fuel_pathway_data", "molecular_info.json"
        )
    with open(filepath, "r") as f:
        return json.load(f)


def load_technology_info(filepath=None):
    if filepath is None:
        filepath = os.path.join(
            top_dir, "input_fuel_pathway_data", "technology_info.json"
        )
    with open(filepath, "r") as f:
        return json.load(f)


def load_global_parameters(filepath=None):
    if filepath is None:
        filepath = os.path.join(
            top_dir, "input_fuel_pathway_data", "global_parameters.json"
        )
    with open(filepath, "r") as f:
        return json.load(f)

