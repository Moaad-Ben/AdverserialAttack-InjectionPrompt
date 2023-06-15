__author__ = "Moaad Ben Amar"

import multiprocessing

from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import utils

logger = None


def wrapperSetupWithHuggingFace(modelName):
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    return HuggingFaceModelWrapper(model, tokenizer)


def loadDataSetFromHuggingFace(dataSetName, split="test"):
    return load_dataset(dataSetName, split=split)


def attack_print(attackResult, attackTimer):
    typeOfAttack = type(attackResult)
    print(f"AttackResultType: {typeOfAttack}")
    print(f"Attack result: {attackResult}")
    print(f"Perturbed Text: {attackResult.perturbed_result.attacked_text}")
    print(attackTimer.timePastAsStr() + " sec")
    print()


def attack_process(singleDataSet, attackRecipe):
    returnBool = False
    text = singleDataSet["text"]
    trueLabel = singleDataSet["label"]
    attackTimer = utils.Timer()
    attackResult = attackRecipe.attack(text, trueLabel)

    if isinstance(attackResult, SuccessfulAttackResult):
        returnBool = True
    elif isinstance(attackResult, SkippedAttackResult):
        returnBool = False
    print(f"Original Text: {text}")
    print(f"Original Label: {trueLabel}")
    attack_print(attackResult, attackTimer)

    return returnBool


def main():
    modelSetupTimer = utils.Timer()
    # Step 1: Load the target model
    modelName = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    targetModel = HuggingFaceModelWrapper(model, tokenizer)

    # Step 2: Load a dataset to test the model
    dataset = load_dataset("imdb", split="test")

    # Step 3: Choose an attack recipe
    attackRecipe = TextFoolerJin2019.build(model_wrapper=targetModel)
    print(modelSetupTimer.timePastAsStr() + " seconds for the model-setup")
    # Step 4: Iterate over the dataset and perform the attack
    successfulAttacks = 0
    skippedAttacks = 0
    totalAttacks = 0
    totalTimer = utils.Timer()

    for tmpDataSet in dataset:

        text = tmpDataSet["text"]
        trueLabel = tmpDataSet["label"]
        attackTimer = utils.Timer()
        totalAttacks += 1
        # Create the attack object
        attackResult = attackRecipe.attack(text, trueLabel)

        print(f"Original Text: {text}")
        print(f"Original Label: {trueLabel}")

        if isinstance(attackResult, SuccessfulAttackResult):
            successfulAttacks += 1

            attack_print(attackResult, attackTimer)

        elif isinstance(attackResult, SkippedAttackResult):
            skippedAttacks += 1
            attack_print(attackResult, attackTimer)

        if successfulAttacks >= 50:
            break

    print("DONE Time for the whole Attack Process:" + totalTimer.timePastAsStr("min"))
    print(f"SkippedAttacks {skippedAttacks}/{totalAttacks}")
    print(f"SuccessfulAttack {successfulAttacks}/{totalAttacks}")


if __name__ == "__main__":
    main()
