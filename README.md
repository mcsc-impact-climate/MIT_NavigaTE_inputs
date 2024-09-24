# MIT Input Files for the NavigaTE Shipping Model
This repo contains input files and processing+analysis code for MIT's use of the NavigaTE maritime shipping model developed by the Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.

# Pre-requisites
Python3 installation

# Instructions for use
## Install NavigaTE
Assuming you have access to the NavigaTE repo, start by cloning the repo

```bash
cd ..
git clone git@github.com:zerocarbonshipping/NavigaTE.git
cd NavigaTE
git checkout MIT-shipping-study
```

Install NavigaTE using the wheel file:
```bash
python setup.py bdist_wheel
pip install -U --no-deps --force-reinstall dist/navigate-4.0.2-py3-none-any.whl
```

Switch back into this repo:

```bash
cd ../MIT_NavigaTE_inputs
```

## Run NavigaTE for a specific pathway 

NavigaTE inputs for the global fleet of container, bulk, tanker and gas carrier vessels are defined in `single_pathway_full_fleet` and `includes_global`.

To run the NavigaTE model for a single fuel production pathway, eg. LSFO:

```bash
navigate --suppress-plots single_pathway_full_fleet/lsfo/lsfo.nav
```

This should produce an output report (`lsfo_pathway_excel_report.xlsx`) in `single_pathway_full_fleet/lsfo_pathway/plots/`.

## Run NavigaTE for all pathways (global fleet)

### Create cost and emissions info for fuel production pathways and processes

The script [`calculate_fuel_costs_emissions.py`](./source/calculate_fuel_costs_emissions.py) leverages the input data file [`fuel_production_inputs.csv`](./input_fuel_pathway_data/fuel_production_inputs.csv) to produce csv files with cost and emissions info relating to fuel production and subsequent downstream processes (hydrogen to ammonia, hydrogen compression, and hydrogen liquefaction). To run:

```bash 
python source/calculate_fuel_costs_emissions.py
```

Currently, this creates the following csv files:
* `input_fuel_pathway_data/production/{fuel}_costs_emissions.csv`: Costs and emissions associated with production of the given `{fuel}`.
*  `input_fuel_pathway_data/process/hydrogen_{process}_costs_emissions.csv`: Costs and emissions associated with the given `{process}`.

### 

The script [`make_cost_emissions_files.py`](./source/make_cost_emissions_files.py) uses cost and emission data for a range of hydrogen and ammonia production pathways and regions in [`input_fuel_pathway_data`](./input_fuel_pathway_data) to generate a set of cost and emission inputs in [`includes_global/all_costs_emissions`](./includes_global/all_costs_emissions). This script is integrated into [`run_all_pathways.sh`](./run_all_pathways.sh), which runs NavigaTE for the global fleet over all available fuel production pathways and regions.

To just make the input `*.inc` files for each pathway:

```bash
python source/make_cost_emissions_files.py
```

This should populate `includes_global/all_costs_emissions`, with one `*.inc` file per fuel production pathway and region.

To run NavigaTE over all pathways and regions:

```bash
bash run_all_pathways.sh
```

This should produce `.xlsx` files in all_outputs_full_fleet with NavigaTE outputs for each fleet and vessel type. 

## Organize NavigaTE outputs

The script [`make_output_csvs.py`](./source/make_output_csvs.py) reads in the `.xslx` files produced by `run_all_pathways.sh`, evaluates some derived quantities, and organizes all the data into csv files in [`processed_results`](./processed_results), with one csv file per country, pathway and evaluation option. The rows represent the production region, and columns represent vessel classes in the global fleet.

To run:

```bash
python source/make_output_csvs.py
```

The naming convention for the csv files is: `{fuel}-{pathway_type}-{pathway}_{quantity}_{evaluation_option}.csv`. Information about each `quantity` can be found in [`info_files/quantity_info.csv`](./processed_results/quantity_info.csv).

The different evaluation_options are defined as follows:
* `"vessel"`: The quantity is per-vessel annually, and unchanged relative to its description in `quantity_info.csv`.
* `"fleet"`: The quantity is aggregated over all global vessels in given vessel class (still evaluated annually).
* `"per_mile"`: The quantity is normalized by the number of annual nautical miles (nm) traveled by the given vessel class.
* `"per_tonne_mile"`: The quantity is normalized by the number of annual tonne-nm carried by the given vessel class.

## Make validation plots

The script [`make_validation_plots.py`](./source/make_validation_plots.py) reads in the processed csv files produced by `make_output_csvs.py` to create a set of validation plots. 

To run:

```bash
python source/make_validation_plots.py
```

This should produce a range of validation plots in the [`plots`](./plots) directory for different fuels, pathways and quantities evaluated by `make_output_csvs.py`. 
