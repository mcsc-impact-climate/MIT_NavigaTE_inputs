# Run with original tank and cargo capacities
#bash run_all_pathways.sh -o -k liquid_hydrogen-LTE_H_solar_E -l orig_caps
bash run_all_pathways.sh -o -k compressed_hydrogen-LTE_H_solar_E -l orig_caps -x
bash run_all_pathways.sh -o -k ammonia-LTE_H_solar_E -l orig_caps -x
#bash run_all_pathways.sh -o -k methanol-LTE_H_DAC_C_solar_E -l orig_caps -x
#bash run_all_pathways.sh -o -k FTdiesel-LTE_H_DAC_C_solar_E -l orig_caps -x
#bash run_all_pathways.sh -o -k lng-fossil -l orig_caps -x

# Run with modified tank and cargo capacities
#bash run_all_pathways.sh -k liquid_hydrogen-LTE_H_solar_E -l mod_caps
bash run_all_pathways.sh -k compressed_hydrogen-LTE_H_solar_E -l mod_caps -x
bash run_all_pathways.sh -k ammonia-LTE_H_solar_E -l mod_caps -x
#bash run_all_pathways.sh -k methanol-LTE_H_DAC_C_solar_E -l mod_caps -x
#bash run_all_pathways.sh -k FTdiesel-LTE_H_DAC_C_solar_E -l mod_caps -x
#bash run_all_pathways.sh -k lng-fossil -l mod_caps -x

python source/make_output_csvs.py -i orig_caps -o orig_caps_no_boiloff
python source/make_output_csvs.py -i mod_caps -o mod_caps_no_boiloff
python source/make_output_csvs.py -b -i mod_caps -o mod_caps_with_boiloff
python source/modify_tanks_and_cargo_capacity.py
python source/analyze_boiloff_tank_size_impact.py
python source/analyze_vessel_design_cargo_displacement_impact.py
