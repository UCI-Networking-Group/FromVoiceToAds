from llm_utility import *
import os
import sys
import pandas as pd


def get_list_categories_from_amazon():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    list_cate = read_from_text_file(os.path.join(file_dir, "amazon_ads_categories.txt")).split('\n')
    return list_cate


def extract_categories_from_ads_file(input_file_path, output_folder_path):
    input_file = os.path.basename(input_file_path)
    cate_file_path = os.path.join(output_folder_path, f"{os.path.splitext(input_file)[0]}_categories.txt")
    ads = read_from_text_file(input_file_path)
    if ads == 'None':
        save_to_text_file('None', cate_file_path)
    else:
        list_cate = get_list_categories_from_amazon()  # change the taxonomy if needed
        res = identify_ads_categories(ads, list_cate)
        save_to_text_file(res, cate_file_path)


def extract_categories_from_ads_files_in_folder(input_folder_path, output_folder_path):
    # Check if input folder exists
    if not os.path.isdir(input_folder_path):
        print(f"Error: Input folder '{input_folder_path}' not found!")
        return
    # Create the output folder if it doesn't exist
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)

    # Categorize all ads files in the input folder
    for file_name in os.listdir(input_folder_path):
        if file_name.lower().endswith('_ads.txt'):
            file_cate = file_name[:-4] + '_categories.txt'
            if os.path.isfile(os.path.join(output_folder_path, file_cate)):
                print(f"Done: {file_name}")
            else:
                input_file_path = os.path.join(input_folder_path, file_name)
                extract_categories_from_ads_file(input_file_path, output_folder_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        script_name = os.path.basename(__file__)
        print(f"Usage: python {script_name} input_(file or folder)_path output_folder_path")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isdir(output_path):
        print(f"Invalid or non-existing output folder: {output_path}")
        sys.exit(1)

    if os.path.isfile(input_path):
        extract_categories_from_ads_file(input_path, output_path)
    elif os.path.isdir(input_path):
        extract_categories_from_ads_files_in_folder(input_path, output_path)
    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)
