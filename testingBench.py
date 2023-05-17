__author__ = "Moaad Ben Amar"

import requests

API_URL = "https://api-inference.huggingface.co/models/winglian/llama-adapter-13b"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    return response.json()


output = query({
    "inputs": {
        "past_user_inputs": ["Which movie is the best ?"],
        "generated_responses": ["It's Die Hard for sure."],
        "text": "Can you explain why ?"
    },
})

print(output)