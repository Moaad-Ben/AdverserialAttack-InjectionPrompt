__author__ = "Moaad Ben Amar"

import logging
import multiprocessing
import time
from logging.handlers import RotatingFileHandler

import textattack
from textattack import Attack
from textattack.attack_recipes import MorpheusTan2020, Seq2SickCheng2018BlackBox
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.goal_functions import MinimizeBleu
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapChangeLocation
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartModel, BartTokenizer, GPT2Model, \
    GPT2Tokenizer, GPT2LMHeadModel, BartForConditionalGeneration, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, \
    T5Tokenizer

from datasets import load_dataset

from CustomModelWrapper import CustomT5ModelWrapper, CustomGPT4AllModelWrapper
import utils


def createTargetModel(model_name, taskType):
    """
    Create the target model based on the given model name and task type.

    Args:
        model_name (str): Name of the model.
        taskType (str): Type of the task.

    Returns:
        ModelWrapper: Wrapper for the target model.
    """

    if model_name == "bert-base-uncased":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HuggingFaceModelWrapper(model, tokenizer)

    elif model_name == "facebook/bart-large":
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        return HuggingFaceModelWrapper(model, tokenizer)

    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return HuggingFaceModelWrapper(model, tokenizer)

    elif model_name == "t5-large":
        return CustomT5ModelWrapper(model_name)

    elif model_name == "bert2bert":
        tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", model_max_length=512)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")
        return HuggingFaceModelWrapper(model, tokenizer)

    elif model_name == "t5-base":
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        return HuggingFaceModelWrapper(model, tokenizer)

    elif model_name == "GPT4All":
        return CustomGPT4AllModelWrapper(taskType)

    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")


def createTextAttack_Bleu_Greedy(targetModel):
    """
    Create a TextAttack attack using the Bleu-Greedy approach.

    Args:
        targetModel (ModelWrapper): Wrapper for the target model.

    Returns:
        Attack: TextAttack attack instance.
    """
    # translation
    transformation = WordSwapChangeLocation()
    searchMethod = GreedySearch()
    constraints = [RepeatModification(), StopwordModification()]
    goalFunction = MinimizeBleu(targetModel)
    return Attack(goalFunction, constraints, transformation, searchMethod)


def attack_print(attackResult, attackTimer):
    """
    Print the attack result and relevant information.

    Args:
        attackResult (AttackResult): Result of the attack.
        attackTimer (Timer): Timer object for attack timing.
    """
    typeOfAttack = type(attackResult)
    print(f"AttackResultType: {typeOfAttack}")
    print(f"Attack result: {attackResult}")
    print(f"Perturbed Text: {attackResult.perturbed_result.attacked_text}")
    print(attackTimer.timePastAsStr() + " sec")
    print()


def tryInjectionAttack(inputText, expectedText, injectionType, target_model):
    """
    Perform the injection attack on the target model.

    Args:
        inputText (str): Input text for the attack.
        expectedText (str): Expected text for comparison.
        injectionType (str): Type of injection attack.
        target_model (ModelWrapper): Wrapper for the target model.
    """
    if injectionType == "Insert String":
        intro = "Insert the following string:"
        injectionLst = utils.loadTxtAsLst("Insert_String")
    elif injectionType == "Change Language":
        intro = "Translate this into"
        injectionLst = utils.loadTxtAsLst("Change_Language")
    elif injectionType == "Jailbreak":
        intro = ""
        injectionLst = utils.loadTxtAsLst("Jailbreak")
    else:
        raise RuntimeError("Error in tryInjectionAttack")

    for injectionPrompt in injectionLst:
        attackPrompt = intro + injectionPrompt
        response = target_model.chatResponse(attackPrompt + inputText)

        print(f"The used AttackPrompt:{attackPrompt} and the InputText:{inputText}")
        print(f"The response of the Model:{response}")
        print(f"This was the expected Text{expectedText}")


def main():
    """
    Main function to execute the attack and evaluation.
    """
    # Settings
    targetNumber = 5
    model_name = "GPT4All"
    # "bert2bert" # "facebook/bart-large" # "t5-base"  # "t5-large"  # "facebook/bart-large" # "google/pegasus-gigaword" # "gpt2"  # "bert-base-uncased"
    taskType = "Text Summarization"  # Machine Translation /Text Summarization/ Text Classification/ Question Answering
    injectionType = "Insert String"  # Insert String /Change Language /Jailbreak

    # Step 1: Initialize the logger
    setupTimer = utils.Timer()
    target_model = createTargetModel(model_name, taskType)

    # Step 3: Load a dataset to test the model
    if taskType == "Text Summarization":
        dataset = load_dataset("gigaword", split="test")
    elif taskType == "Question Answering":
        dataset = load_dataset("squad_v2", split="validation")
    elif taskType == "Machine Translation":
        dataset = load_dataset("wmt14", "de-en", split="test")
        transformedDataset = [(item["translation"]["en"], item["translation"]["de"]) for item in dataset]
        dataset = textattack.datasets.Dataset(transformedDataset)
    elif taskType == "Text Classification":
        dataset = load_dataset("imdb", split="test")
    else:
        raise NotImplementedError()

    if injectionType == "No Injection":
        # Step 4: Choose an attack recipe

        # attack = TextFoolerJin2019.build(model_wrapper=target_model)
        # attack = TextBuggerLi2018.build(model_wrapper=target_model)
        attack = Seq2SickCheng2018BlackBox.build(model_wrapper=target_model)
        # attack = MorpheusTan2020.build(model_wrapper=target_model)

        print(f"Setup took " + setupTimer.timePastAsStr() + " seconds")

        attackArgs = textattack.AttackArgs(
            num_successful_examples=targetNumber,
            shuffle=True,
            parallel=False,
            log_to_txt=f"LOGS/automatic AttackLog for {taskType}",
            enable_advance_metrics=True
        )

        attacker = textattack.Attacker(attack, dataset, attack_args=attackArgs)
        attacker.attack_dataset()

    else:
        totalAttacks = 0
        successfulResponse = 0
        skippedResponse = 0
        bleu_greedy_attack = createTextAttack_Bleu_Greedy(target_model)
        for datasetEntry in dataset:
            if taskType == "Text Summarization":
                inputText = datasetEntry["document"]  # whole text
                expectedText = datasetEntry["summary"]  # summary

            elif taskType == "Machine Translation":
                inputText = datasetEntry[0]["text"]  # english text
                expectedText = datasetEntry[1]  # german translation
            else:
                raise RuntimeError("Task Type is not supported or was written wrong for Prompt Injection Attack.")

            attackTimer = utils.Timer()
            totalAttacks += 1
            attackResult = bleu_greedy_attack.attack(inputText, expectedText)
            if isinstance(attackResult, SkippedAttackResult):
                skippedResponse += 1
                attack_print(attackResult, attackTimer)

            else:
                successfulResponse += 1
                attack_print(attackResult, attackTimer)
                tryInjectionAttack(inputText, expectedText, injectionType, target_model)


if __name__ == "__main__":
    main()
