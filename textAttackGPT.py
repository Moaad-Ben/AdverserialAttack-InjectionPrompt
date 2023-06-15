__author__ = "Moaad Ben Amar"

import logging
import multiprocessing
from logging.handlers import RotatingFileHandler

import textattack
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019, TextBuggerLi2018
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as tflogging
from datasets import load_dataset
import utils
from testingBench import attackTheModelMP





class TextAttackLogger:
    def __init__(self, log_file="LOGS/AdversarialAttack"):
        self.logger = tflogging.get_logger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        # Create a StreamHandler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Create a FileHandler to write logs to a file
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def log_attack_result(self, attack_result, attack_timer):
        type_of_attack = type(attack_result)
        perturbed_text = attack_result.perturbed_result.attacked_text
        perturbed_label = attack_result.perturbed_result.attacked_text

        print(f"AttackResultType: {type_of_attack}")
        print(f"Attack result: {attack_result}")
        print(f"Perturbed Text: {perturbed_text}")
        print(f"Perturbed Label: {perturbed_label}")
        print(attack_timer.timePastAsStr() + " sec")
        print()

        self.log(f"AttackResultType: {type_of_attack}")
        self.log(f"Attack result: {attack_result}")
        self.log(f"Perturbed Text: {perturbed_text}")
        self.log(f"Perturbed Label: {perturbed_label}")
        self.log(attack_timer.timePastAsStr() + " sec")


def main():
    targetNumber = 100
    # attack_logger = TextAttackLogger()
    # # Configure the transformers logger
    # logging.basicConfig(level=logging.INFO)
    # transformers_logger = logging.getLogger("transformers")
    # transformers_logger.setLevel(logging.INFO)
    # logger = TextAttackLogger('attack.log')

    # Step 1: Initialize the logger
    setupTimer = utils.Timer()
    # Step 2: Load the target model
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_model = HuggingFaceModelWrapper(model, tokenizer)

    # Step 3: Load a dataset to test the model
    dataset = load_dataset("imdb", split="test")

    # Step 4: Choose an attack recipe
    attackRecipe = TextBuggerLi2018.build(model_wrapper=target_model)
    # logger.log(setupTimer.timePastAsStr() + " seconds for the model-setup")
    print(f"Setup took " + setupTimer.timePastAsStr() + " seconds")
    # Step 5: Iterate over the dataset and perform the attack
    successfulAttacks = 0
    skippedAttacks = 0
    totalAttacks = 0
    totalTimer = utils.Timer()

    multiprocessing.set_start_method('spawn')

    while successfulAttacks <= targetNumber:
        processes = []
        for i in range(4):
            print("Starting Attack Nr." + str(totalAttacks))
            tmpDataset = dataset[totalAttacks]
            process = multiprocessing.Process(target=attackTheModelMP, args=(tmpDataset, attackRecipe))
            processes.append(process)
            process.start()

            totalAttacks += 1

        for process in processes:
            process.join()

        successfulAttacks += sum(
            [process.exitcode == 0 and process.exitcode is not None and process.exitcode != 1 for process in processes])

    print("DONE Time for the whole Attack Process:" + totalTimer.timePastAsStr("min") + " min")
    print(f"SkippedAttacks {skippedAttacks}/{totalAttacks}")
    print(f"SuccessfulAttack {successfulAttacks}/{totalAttacks}")


if __name__ == "__main__":
    main()

# def attack_process(singleDataSet, attackRecipe, logger):
#     returnBool = False
#     text = singleDataSet["text"]
#     trueLabel = singleDataSet["label"]
#     attackTimer = utils.Timer()
#     attackResult = attackRecipe.attack(text, trueLabel)
#
#     if isinstance(attackResult, SuccessfulAttackResult):
#         returnBool = True
#     elif isinstance(attackResult, SkippedAttackResult):
#         returnBool = False
#
#     logger.log(f"Original Text: {text}")
#     logger.log(f"Original Label: {trueLabel}")
#     attack_print(attackResult, attackTimer, logger)
#
#     return returnBool
