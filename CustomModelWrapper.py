__author__ = "Moaad Ben Amar"

from gpt4all import gpt4all
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class CustomT5ModelWrapper:
    def __init__(self, model_name_or_path):
        # Initialize the CustomT5ModelWrapper
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, model_max_length=512)

    def __call__(self, text_input_list):
        # Call method to generate texts based on input list
        input_ids = self.tokenizer.batch_encode_plus(
            text_input_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=50)

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts


class CustomGPT4AllModelWrapper:
    # Models can be found in the folder: C:/Users/{windowsUser}/AppData/Local/nomic.ai/GPT4All/
    def __init__(self, taskType, attackType=None):
        # Initialize the CustomGPT4AllModelWrapper
        availableModels = [
            "ggml-gpt4all-j-v1.3-groovy",
            "ggml-gpt4all-l13b-snoozy",
            "ggml-mpt-7b-chat",
            "ggml-gpt4-x-alpaca-13b-native-4bit-128g"
        ]
        self.gptj = gpt4all.GPT4All(model_name=availableModels[3],
                                    model_path="C:/Users/benam/AppData/Local/nomic.ai/GPT4All/",
                                    model_type="llama", allow_download=False)
        self.model = self.gptj.model
        self.tokenizer = None
        self.taskType = taskType
        print(f"GPT4All is setup with the model: {self.gptj.model}")

    def __call__(self, text_input):
        # Call method to process text input
        prompt = self.createTaskPrompt()
        if isinstance(text_input, list):
            nrOfInputs = len(text_input)
            if nrOfInputs == 1:
                return self.responseCatching(prompt + str(text_input))

            elif nrOfInputs > 1:
                responseLst = []
                for item in text_input:
                    modelInput = prompt + item
                    responseLst.append(self.responseCatching(modelInput))
                return responseLst
            else:
                raise RuntimeError("text_input has no entries!")

    def createTaskPrompt(self):
        # Create the task prompt based on the task type
        prompt = ""
        if self.taskType == "No Attack":
            pass
        elif self.taskType == "Question Answering":  # Context has to be given not jet implemented
            raise NotImplementedError("Context has to be given")
            # prompt = f"Please answer the following question based on the given context: Context: {context} Question: "
        elif self.taskType == "Machine Translation":
            prompt = "Translate this into german: "
        elif self.taskType == "Text Summarization":
            prompt = "Provide a summary of the following text: "
        elif self.taskType == "Text Classification":
            prompt = "Classify the sentiment of the following text by assigning it a label of either 1 (positive) or 0 (negative): Text: "
        else:
            raise RuntimeError("No Type of Attack specified!")
        return prompt

    def responseCatching(self, text_input):
        # Catch the response from the GPT4All model
        messages = [{"role": "user", "content": text_input}]
        response = self.gptj.chat_completion(messages, verbose=True, default_prompt_header=False,
                                             default_prompt_footer=False)
        responseMsg = response["choices"][0]["message"]["content"]
        responseMsg = responseMsg.encode('ascii', 'ignore').decode('ascii')  # to prevent russian and chinese
        return responseMsg

    def chatResponse(self, text_input):
        # Get the response from the GPT4All model for chat-based input
        prompt = self.createTaskPrompt()
        return self.responseCatching(prompt + text_input)
