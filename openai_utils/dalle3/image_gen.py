import os

from openai import OpenAI
import textwrap
import requests

with open('openai_utils/openai-api.key', 'r') as f:
    api_key = f.read()

client = OpenAI(api_key=api_key)

def process_response(response, file_name=None):
    file_name = file_name or 'dalle3_image.jpg'
    for i, d in enumerate(response.data):
        display_data(d, file_name= f"{i}_{file_name}")

def display_data(data, file_name=None):
    print("Revised prompt:")
    print(textwrap.fill(data.revised_prompt, width=80))
    print()
    print("Image url:")
    print(data.url)
    # get cur dir
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{cur_dir}/dalle_images/{file_name}", 'wb') as f:
        f.write(requests.get(data.url).content)
    print(f"Image saved to {file_name}")


if __name__ == '__main__':

    prompt = "Depiction of Chirag Falor"
    file_name = "cfalor.jpeg"

    query_args = {
        "model": "dall-e-3",
        "size": "1024x1024",
        "quality": "standard",
        "prompt": prompt,
        "n": 1,
    }


    response = client.images.generate(**query_args)
    process_response(response, file_name=file_name)