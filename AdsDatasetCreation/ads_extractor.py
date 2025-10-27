from llm_utility import *
import re
import sys
import os


def parse_advertisements(content):
    # Define the keywords for start and end markers
    start_keyword = "Advertisements"
    end_keyword = "Other Content"
    # Use regular expressions to find the markers dynamically
    start_marker = re.search(f'{re.escape(start_keyword)}:(.*?)(?={re.escape(end_keyword)}:|$)', content, re.DOTALL)
    end_marker = re.search(f'{re.escape(end_keyword)}:(.*?)(?={re.escape(start_keyword)}:|$)', content, re.DOTALL)
    if start_marker and end_marker:
        # Extract and clean up the advertisements
        advertisements_text = start_marker.group(1).strip()
        # Remove lines with markers
        advertisements_text = '\n'.join(line.strip() for line in advertisements_text.split('\n') if
                                        not line.strip().startswith(start_keyword) and not line.strip().startswith(
                                            end_keyword))
        return advertisements_text
    else:
        print("Start or end marker not found.")
        return []


def extract_ads_from_transcript_file(input_file_path, output_folder_path):
    transcript = read_from_text_file(input_file_path)

    input_file = os.path.basename(input_file_path)

    # Extract ads
    ads_file_path = os.path.join(output_folder_path, f"{os.path.splitext(input_file)[0][:-11]}_ads.txt")
    ads = extract_ads(transcript)
    save_to_text_file(ads, ads_file_path)

    # Get ads products
    #ads_products_file_path = os.path.join(output_folder_path, f"{os.path.splitext(input_file)[0]}_ads_products.txt")
    #if ads == 'None':
    #    save_to_text_file('None', ads_products_file_path)
    #else:
    #    ads_products = identify_ads_products(ads)
    #    save_to_text_file(ads_products, ads_products_file_path)


def extract_ads_from_transcript_files_in_folder(input_folder_path, output_folder_path):
    # Check if input folder exists
    if not os.path.isdir(input_folder_path):
        print(f"Error: Input folder '{input_folder_path}' not found!")
        return
    # Create the output folder if it doesn't exist
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)

    # Extract ads from all files in the input folder
    for file_name in os.listdir(input_folder_path):
        if file_name.lower().endswith('_transcript.txt'):
            file_ads = file_name[:-15] + '_ads.txt'
            if os.path.isfile(os.path.join(output_folder_path, file_ads)):
                print(f"Done: {file_name}")
            else:
                input_file_path = os.path.join(input_folder_path, file_name)
                extract_ads_from_transcript_file(input_file_path, output_folder_path)


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
        extract_ads_from_transcript_file(input_path, output_path)
    elif os.path.isdir(input_path):
        extract_ads_from_transcript_files_in_folder(input_path, output_path)
    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)
