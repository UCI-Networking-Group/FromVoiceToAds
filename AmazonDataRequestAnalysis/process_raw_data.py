import os
import csv
import json


def collect_csv_data(folder_path, target_file):
    """
    Recursively collect data from all 'name.csv' files in the given folder and its subfolders.
    """
    csv_data = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(target_file):
                file_path = os.path.join(root, file_name)
                with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader, None)  # Skip the header
                    # Append the single-column values to the list
                    csv_data.extend(row[0] for row in reader)
    return csv_data


def process_csv_files_with_nested(root_dir):
    """
    Create a JSON dictionary with first-level subfolder names as keys and data from 'name.csv' files as values.
    """
    result = {}
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Collect all CSV data from this subfolder and its nested subfolders
            result[subfolder] = collect_csv_data(subfolder_path)
    return result


if __name__ == "__main__":
    raw_data_folder = "AmazonDataRequest"

    # Process AdvertiserAudiences
    output_file = "AdvertiserAudiences.json"
    json_data = process_csv_files_with_nested(raw_data_folder, "Advertising.AdvertiserAudiences.csv")
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"JSON data has been saved to {output_file}")

    # Process AmazonAudiences
    output_file = "AmazonAudiences.json"
    json_data = process_csv_files_with_nested(raw_data_folder, "Advertising.AmazonAudiences.csv")
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"JSON data has been saved to {output_file}")