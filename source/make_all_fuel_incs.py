"""
Date: May 21, 2025
Purpose: Create fuel include files for all modelled pathways
"""

import os
import re
from collections import defaultdict
from common_tools import get_top_dir

top_dir = get_top_dir()

def create_all_pathways_includes(all_costs_dir: str, includes_dir: str) -> None:
    """
    Generate {fuel}-all_pathways.inc files for each unique fuel, collecting
    process variants from all_costs_emissions/ and using the original fuel
    definitions in includes_global/.

    Args:
        all_costs_dir (str): Directory containing input .inc files per fuel-pathway-country.
        includes_dir (str): Directory containing base {fuel}.inc and target output files.
    """

    # Mapping from fuel to set of processes
    fuel_processes = defaultdict(set)

    # Extract fuel and process from filenames
    pattern = re.compile(r'(?P<fuel>[^-]+)-(?P<process>.+)-[^-]+-1\.inc')

    for fname in os.listdir(all_costs_dir):
        if fname.endswith('.inc'):
            match = pattern.match(fname)
            if match:
                fuel = match.group('fuel')
                process = match.group('process')
                fuel_processes[fuel].add(process)

    # Generate one file per fuel
    for fuel, processes in fuel_processes.items():
        if fuel.startswith("compressed"):
            continue
        base_fuel_path = os.path.join(includes_dir, f"{top_dir}/includes_global/{fuel}.inc")
        output_path = os.path.join(includes_dir, f"{top_dir}/includes_global/{fuel}-all_pathways.inc")

        if not os.path.exists(base_fuel_path):
            print(f"Skipping {fuel}: missing {fuel}.inc")
            continue

        # Read and parse original fuel definition file
        with open(base_fuel_path, 'r') as f:
            lines = f.readlines()

        header_comments = []
        fuel_block = []
        in_fuel_block = False
        for line in lines:
            if line.strip().startswith(f'Fuel "{fuel}"'):
                in_fuel_block = True
            if not in_fuel_block:
                header_comments.append(line)
            else:
                fuel_block.append(line)

        # Convert "Fuel "{fuel}" {" → "Fuel "{fuel}-*" {"
        wildcard_block = ''.join(fuel_block).replace(f'Fuel "{fuel}"', f'Fuel "{fuel}-*"', 1)

        # Write new file
        with open(output_path, 'w') as f_out:
            f_out.writelines(header_comments)
            f_out.write("\n")

            for proc in sorted(processes):
                f_out.write(f'Fuel "{fuel}-{proc}" {{\n}}\n')
            f_out.write("\n")
            f_out.write(wildcard_block)

        print(f"Created {output_path}")
        
import os
import re
from collections import defaultdict

def get_fuel_processes(all_costs_dir: str) -> dict:
    """
    Scans the directory for .inc files and returns a mapping from fuel to a set of unique process strings,
    skipping any fuels whose name starts with 'compressed'.

    Args:
        all_costs_dir (str): Directory containing {fuel}-{process}-{country}-1.inc files.

    Returns:
        dict: Dictionary mapping each fuel to a set of process names.
    """
    fuel_processes = defaultdict(set)
    pattern = re.compile(r'(?P<fuel>[^-]+)-(?P<process>.+)-[^-]+-1\.inc')

    for fname in os.listdir(all_costs_dir):
        if not fname.endswith('.inc'):
            continue
        match = pattern.match(fname)
        if match:
            fuel = match.group('fuel')
            if fuel.startswith('compressed'):
                continue
            process = match.group('process')
            fuel_processes[fuel].add(process)

    return fuel_processes

        
def generate_port_cost_emissions_files(port_info: dict, fuel_processes: dict, top_dir: str) -> None:
    """
    For each fuel and set of ports, create cost and emissions overwrite files for ports using
    country-specific .inc files.

    Args:
        port_info (dict): Dictionary with keys 'countries' and 'ports', each a list of same length.
        fuel_processes (dict): Dictionary from fuel to set of pathway strings.
        top_dir (str): Base directory path.
    """
    assert len(port_info['countries']) == len(port_info['ports']), "countries and ports lists must be same length"

    includes_dir = os.path.join(top_dir, 'includes_global')
    all_costs_dir = os.path.join(includes_dir, 'all_costs_emissions')

    for fuel, processes in fuel_processes.items():
        filename = f"cost_emissions_{fuel}-{port_info['ports'][0]}_{port_info['ports'][1]}.inc"
        output_path = os.path.join(includes_dir, filename)

        port_blocks = []

        for country, port in zip(port_info['countries'], port_info['ports']):
            port_block = [f'Port "port_{port}" {{\n', '    \n', '    # Emissions\n']

            for process in sorted(processes):
                inc_path = os.path.join(all_costs_dir, f"{fuel}-{process}-{country}-1.inc")
                if not os.path.exists(inc_path):
                    print(f"Warning: missing file {inc_path}")
                    continue

                with open(inc_path, 'r') as f:
                    content = f.read()

                # Extract emissions and cost values
                emission_match = re.search(r'set_bunker_WTT_overwrite\("[^"]+",\s*"carbon_dioxide",\s*([-0-9.]+)\)', content)
                cost_match = re.search(r'set_bunker_price_overwrite\("[^"]+",\s*([-0-9.]+)\)', content)

                if emission_match and cost_match:
                    emission_val = float(emission_match.group(1))
                    cost_val = float(cost_match.group(1))
                    port_block.append(f'    set_bunker_WTT_overwrite("{fuel}-{process}", "carbon_dioxide", {emission_val})\n')
                else:
                    print(f"Warning: could not find values in {inc_path}")
                    continue

            port_block.append('\n    # Costs\n')
            for process in sorted(processes):
                inc_path = os.path.join(all_costs_dir, f"{fuel}-{process}-{country}-1.inc")
                if not os.path.exists(inc_path):
                    continue

                with open(inc_path, 'r') as f:
                    content = f.read()

                cost_match = re.search(r'set_bunker_price_overwrite\("[^"]+",\s*([-0-9.]+)\)', content)
                if cost_match:
                    cost_val = float(cost_match.group(1))
                    port_block.append(f'    set_bunker_price_overwrite("{fuel}-{process}", {cost_val})\n')

            port_block.append('\n}\n\n')
            port_blocks.extend(port_block)

        with open(output_path, 'w') as f_out:
            f_out.writelines(port_blocks)

        print(f"[✓] Created {output_path}")
        
def write_bunker_logistics_file(fuel_processes: dict, includes_dir: str) -> None:
    """
    Writes a bunker_logistics_all-pathways.inc file listing all Fuel("{fuel}-{pathway}") combinations.

    Args:
        fuel_processes (dict): Dictionary mapping fuel → set of pathway strings.
        includes_dir (str): Path to includes_global directory where the output file will be saved.
    """
    output_path = os.path.join(includes_dir, 'bunker_logistics_all_pathways.inc')

    fuel_lines = []
    for fuel, processes in sorted(fuel_processes.items()):
        for process in sorted(processes):
            fuel_lines.append(f'        Fuel("{fuel}-{process}"),')

    # Remove trailing comma from the last entry
    if fuel_lines:
        fuel_lines[-1] = fuel_lines[-1].rstrip(',')

    with open(output_path, 'w') as f_out:
        f_out.write("BunkerLogistics {\n")
        f_out.write("    LiquidMarketFuels = [\n")
        f_out.write('\n'.join(fuel_lines))
        f_out.write("\n    ]\n")
        f_out.write("}\n")

    print(f"[✓] Created {output_path}")



def main():
    all_costs_dir = f'{top_dir}/includes_global/all_costs_emissions'
    includes_dir = f'{top_dir}/includes_global'
    
    # Extract fuel-to-process mappings
    fuel_processes = get_fuel_processes(all_costs_dir)
    
    # Create {fuel}-all_pathways.inc files
    create_all_pathways_includes(all_costs_dir, includes_dir)
    
    # Define port-country mapping
    port_info = {
        "countries": ["Singapore", "Netherlands"],
        "ports": ["singapore", "rotterdam"]
    }
    
    # Create cost/emissions overwrite files per port
    generate_port_cost_emissions_files(port_info, fuel_processes, top_dir)
    
    # Write bunker logistics file
    write_bunker_logistics_file(fuel_processes, includes_dir)

if __name__ == '__main__':
    main()
