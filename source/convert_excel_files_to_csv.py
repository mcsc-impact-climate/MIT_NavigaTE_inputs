"""
Date: 241007
Author: danikam
Purpose: Converts all Excel files produced by NavigaTE into CSV files for further processing.
"""

import argparse
import glob
import os

import pandas as pd


def remove_all_files_in_directory(directory_path):
    """
    Removes all files in the specified directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory where all files will be removed.
    """
    files = glob.glob(os.path.join(directory_path, "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")


def convert_excel_to_csv(input_dir, output_dir):
    """
    Converts the "Vessels" sheet of all Excel files in the input directory to CSV and saves them in the output directory.

    Parameters
    ----------
    input_dir : str
        Directory containing the Excel files.

    output_dir : str
        Directory to save the converted CSV files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove all existing files in the output directory
    remove_all_files_in_directory(output_dir)

    # Get all Excel files from the input directory
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx"))

    # Convert each Excel file to CSV
    for excel_file in excel_files:
        try:
            # Read only the "Vessels" sheet from the Excel file
            df = pd.read_excel(excel_file, sheet_name="Vessels")

            # Create CSV filename with the same name but a .csv extension
            csv_filename = os.path.join(
                output_dir, os.path.basename(excel_file).replace(".xlsx", ".csv")
            )

            # Save the "Vessels" sheet as CSV
            df.to_csv(csv_filename, index=False)

        except Exception as e:
            print(f"Error converting {excel_file}: {e}")
    print(f"Converted excel files in {input_dir} to csv files in {output_dir}")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Convert Excel files to CSV format.")

    # Add input and output directory arguments with --input_dir or -i, and --output_dir or -o
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="The directory containing the Excel files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="The directory to save the converted CSV files.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Convert Excel files to CSV
    convert_excel_to_csv(args.input_dir, args.output_dir)
