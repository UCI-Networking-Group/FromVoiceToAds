from openai import OpenAI
import whisper
import os

try:
    import api_keys
    client = OpenAI(
        api_key=api_keys.OPENAI_API_KEY
    )
except:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )


GPT_MODEL_VERSION = "gpt-4o"



def save_json_string_to_file(json_string, file_path):
    with open(file_path, 'w') as json_file:
        json_file.write(json_string)
    print(f"JSON string written to {file_path}")


def save_to_text_file(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(data)
    print(f"Data written to {file_path}")


def read_from_text_file(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as text_file:
            content = text_file.read()
            return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def transcribe_audio_alt(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return transcription


def transcribe_audio(file_path):
    whisper_model = whisper.load_model("medium.en", device="cuda")
    return whisper_model.transcribe(file_path, condition_on_previous_text=False)["text"]


def identify_ads_products(transcription):
    response = client.chat.completions.create(
        model=GPT_MODEL_VERSION,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following list of advertisements, identify the name of the service or product that was discussed in each item. Give the name only."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


def identify_ads_categories(transcription, list_categories):
    sys_prompt = str(list_categories) + "\nBased on the provided list of categories, identify the categories for the advertisements in the following text and output the advertisement along with its corresponding category in json dictionary format (example: [{'advertisement': advertisement, 'category': category}]). If no category can be identified for an advertisement, use 'Other' category for it."
    response = client.chat.completions.create(
        model=GPT_MODEL_VERSION,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


def extract_ads(transcription):
    response = client.chat.completions.create(
        model=GPT_MODEL_VERSION,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in identifying advertisements in a given text. Separate advertisements appeared in the following text and the other content into 2 lists. Remove repeated phrases that are next to each other and return 'None' if no advertisements were found."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


def analyze_relation(data):
    response = client.chat.completions.create(
        model=GPT_MODEL_VERSION,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in analyzing the relationship between given sets of advertisement topics. Given the following sets of advertisement topics, analyze the similarity and difference between the sets."
            },
            {
                "role": "user",
                "content": data
            }
        ]
    )
    return response.choices[0].message.content
