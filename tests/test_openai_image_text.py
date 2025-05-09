'''
This script demonstrates how to use the OpenAI API (version 0.8.0)
for tasks involving image and text prompts, specifically using a model
that supports vision capabilities (e.g., a GPT-4 vision model).

Make sure you have your OpenAI API key set as an environment variable
named OPENAI_API_KEY, or you can set it directly in the script.
'''
from openai import OpenAI
import base64
import os

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "./tests/img.jpeg"

# Getting the Base64 string
base64_image = encode_image(image_path)

# response = client.responses.create(
#     model="gpt-4.1-mini",
#     input=[{
#         "role": "user",
#         "content": [
#             {"type": "input_text", "text": "what's in this image?"},
#             {
#                 "type": "input_image",
#                 "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#             },
#         ],
#     }],
# )

# print(response.output_text)

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what's in this image?" },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)