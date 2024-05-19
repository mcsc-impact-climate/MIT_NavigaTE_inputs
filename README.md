# MIT Input Files for the NavigaTE Shipping Model
This repo contains input files for MIT's use of the NavigaTE maritime shipping model developed by the Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.

# Pre-requisites
Python3 installation

# Instructions for use
## Install NavigaTE
Assuming you have access to the NavigaTE repo, start by cloning the repo

```bash
cd ..
git clone git@github.com:zerocarbonshipping/NavigaTE.git
cd NavigaTE
git checkout -b [your desired branch, if different from main]
```

Install NavigaTE using the wheel file:
```bash
python setup.py bdist_wheel
pip install -U --no-deps --force-reinstall dist/navigate-4.0.2-py3-none-any.whl
```

Switch back into this repo

## Run NavigaTE
Run the NavigaTE model with a sample `.nav` file, eg.

```bash
cd ../MIT_NavigaTE_inputs
navigate single_pathway_single_vessel/lsfo_pathway/lsfo_pathway.nav
```

This should produce validation plots (`*.png`) and an output report (`lsfo_pathway_excel_report.xlsx`) in `single_pathway_single_vessel/lsfo_pathway/plots`.

