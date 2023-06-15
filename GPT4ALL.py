__author__ = "Moaad Ben Amar"

from abc import ABC

import textattack.models.wrappers.model_wrapper
from gpt4all import GPT4All, gpt4all

# Models can be found in the folder: C:/Users/benam/AppData/Local/nomic.ai/GPT4All/
availableModels = ["ggml-gpt4all-j-v1.3-groovy", "ggml-gpt4all-l13b-snoozy", "ggml-mpt-7b-chat"]


def checkIfModelIsAvailable(model_name):
    if model_name in availableModels:
        return True
    else:
        return False


class GPT4AllModelWrapper(textattack.models.wrappers.model_wrapper.ModelWrapper):

    def __int__(self, model_name):
        self.gpt4allModel = GPT4AllProxy(model_name)

    def __call__(self, text_input_list, **kwargs):
        return self.gpt4allModel.sendMsg("user", text_input_list, **kwargs)


class GPT4AllProxy:

    def __init__(self, model_name):
        if checkIfModelIsAvailable(model_name) is False:
            raise RuntimeError(f"The model with the name {model_name} is not available!")

        self.gptj = gpt4all.GPT4All(model_name=model_name,
                                    model_path="C:/Users/benam/AppData/Local/nomic.ai/GPT4All/",
                                    allow_download=False)

    def sendMsg(self, role, content, **kwargs):
        if not (len(kwargs) == 0):
            print(**kwargs)
        message = [{"role": role, "content": content}]
        return self.gptj.chat_completion(message)
