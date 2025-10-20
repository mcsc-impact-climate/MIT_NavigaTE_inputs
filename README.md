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

## Generate emission rates by commodity

The script [`tabulate_emission_rates_by_commodity.py`](./source/tabulate_emission_rates_by_commodity.py) reads in the per-mile emission rates - well-to-gate (WTT), tank-to-wake (TTW), and well-to-wake (WTW) - output to [`processed_results`](./processed_results) by [`make_output_csvs.py`](./source/make_output_csvs.py) and calculates the corresponding per-tonne-mile and per-cbm-mile (cbm=m<sup>3</sup>) emission rates for each commodity in the U.S. FHWA's Freight Analysis Framework.

Important notes:
* This calculation assumes that the given vessel is filled to its max capacity with the given commodity. For each commodity, the vessel's mass and volume capacity accounts for the commodity's stowage factor (upper/lower bounds and central value).
* For a given commodity, the output files only contain emissions data for the vessel type (container, bulk, tanker, or gas carrier) that is assumed to carry that commodity. 

To execute:

```bash
python source/tabulate_emission_rates_by_commodity.py
```

If there are per-mile csv files for WTT, TTW, and WTW in [`processed_results`](./processed_results), this should produce corresponding per-mile, per-tonne-mile, and per-cbm-mile csv files for the corresponding fuel and production pathway for all commodities in the output directory `emissions_by_commodity`. The output files have the following form:

```bash
{fuel}-{fuel-production-pathway}-{emission_type}-{denominator}_commodity_{commodity}_{sf-option}_sf.csv
```

where:
* `fuel` (eg. `methanol`) is the name of the fuel that the vessel is operating on. The vessel's cargo capacity is adjusted as needed to make space for the fuel tank such that the vessel has the same design range (in nautical miles) as a conventional vessel of the same type and class (eg. bulk_carrier_capesize). 
* `fuel-production-pathway` (eg. `SMR_H_DAC_C_wind_E`) represents the pathway used to produce the fuel. For the example of `SMR_H_DAC_C_wind_E`, the hydrogen (`H`) is produced via steam methane reforming (SMR), carbon (`C`) is produced via direct air capture (`DAC`) and the electricity (`E`) is provided by wind power. Details can be found in [Eamer et al., 2025](https://onepetro.org/SNAMESMC/proceedings/SMC25/SMC25/D031S022R001/792249?searchresult=1).
* `commodity`: The name of the commodity that the vessel is carrying, following the commodities and naming conventions used by the U.S. FHWA's [Freight Analysis Framework](https://ops.fhwa.dot.gov/freight/freight_analysis/faf/) - details can be found in Table 2 of the [FAF5 User Guide](https://www.bts.gov/sites/bts.dot.gov/files/2021-02/FAF5-User-Guide.pdf).
* `emission_type`: can be one of three values:
  * `TotalEquivalentWTT`: This is the **well-to-gate** emissions associated with producing the fuel. Note that while the label "WTT" stands for well-to-tank, this is currently a misnomer, as the calculation of TotalEquivalentWTT doesn't currently include emissions associated with fuel delivery to the port, storage at port, or bunkering, and thus the boundary is not technically well-to-tank. If using grid electricity, the well-to-gate emissions in general depends on the country in which the fuel is produced, which is tabulated in the output file. 
  * `TotalEquivalentTTW`: This is the "tank-to-wake" (TTW) emissions associated with combusting the fuel on the vessel using an internal combustion engine. The TTW emissions depends on the fuel that the vessel is burning, but is independent of the fuel's production pathway.
  * `TotalEquivalentWTW` = `TotalEquivalentWTT+TotalEquivalentTTW`: This is the full well-to-wake emissions including fuel production and combustion (but excluding delivery, storage, and bunkering).
* `denominator`: Can be one of `miles`, `tonne-miles`, or `cbm-miles`. The `tonne-miles` and `cbm-miles` metrics assume that the vessel is filled to its maximum capacity with the given commodity. The capacity may be either mass- or volume-limited depending on the commodity's stowage factor (SF), and this will determine the mass and volume capacity as described in Section 3.5 of [this manuscript](https://www.dropbox.com/scl/fi/3dr02zcofbgz7tllqootp/Cargo_Displacement_Manuscript.pdf?rlkey=4eqysv131j4nkqjovjv7t8n3k&e=1&st=60vw7kel&dl=0).
* `sf-option`: Can be one of `lower`, `upper`, or `central`, where `lower` indicates that the stowage factor is set to its lower bound for the given commodity, `upper` denotes the upper bound, and central denotes the central value (details in Appendix F of [this manuscript](https://www.dropbox.com/scl/fi/3dr02zcofbgz7tllqootp/Cargo_Displacement_Manuscript.pdf?rlkey=4eqysv131j4nkqjovjv7t8n3k&e=1&st=60vw7kel&dl=0)). In all cases except crude petroleum, the central value is the average of the lower and the upper bound. In the case of crude petroleum, the central value is evaluated as the weighted mean over all API ranges, with weights determined by the tonne-miles of crude petroleum produced in the U.S. within a given API range - details can be found in Appendix D of [this manuscript](https://www.dropbox.com/scl/fi/3dr02zcofbgz7tllqootp/Cargo_Displacement_Manuscript.pdf?rlkey=4eqysv131j4nkqjovjv7t8n3k&e=1&st=60vw7kel&dl=0). 

