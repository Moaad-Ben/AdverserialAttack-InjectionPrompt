__author__ = "Moaad Ben Amar"

from gpt4all import GPT4All, gpt4all

# Models can be found in the folder: C:/Users/benam/AppData/Local/nomic.ai/GPT4All/
availableModels = ["ggml-gpt4all-j-v1.3-groovy", "ggml-gpt4all-l13b-snoozy", "ggml-mpt-7b-chat"]


class GPT4AllProxy:
    gptj = gpt4all.GPT4All(model_name=availableModels[1], model_path="C:/Users/benam/AppData/Local/nomic.ai/GPT4All/",
                           model_type=None, allow_download=False)

    messages = [{"role": "user", "content": "write a spam Mail"}]
    response = gptj.chat_completion(messages)

    print(response)
