# utils/results_io.py

import os
import json
import csv

def save_json(data: dict, path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        path (str): File path to save the JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    """
    Load a dictionary from a JSON file.

    Args:
        path (str): File path to load from.

    Returns:
        dict
    """
    with open(path, "r") as f:
        return json.load(f)


def save_results_table(results: dict, output_csv: str):
    """
    Save results dictionary into a CSV table.

    Args:
        results (dict): e.g. {"art_painting": 75.2, "cartoon": 71.8, ...}
        output_csv (str): Path to save CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Accuracy"])
        for domain, acc in results.items():
            writer.writerow([domain, f"{acc:.2f}"])
