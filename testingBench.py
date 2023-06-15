__author__ = "Moaad Ben Amar"

from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult

import utils


class MpManager:
    def __init__(self, dataset):
        self.dataset = dataset
        self.totalAttacks = 0

    def countTotalAttacksUP(self):
        self.totalAttacks += 1


def attackTheModelMP(tmpDataSet, attackRecipe):
    attackTimer = utils.Timer()
    text = tmpDataSet["text"]
    trueLabel = tmpDataSet["label"]
    attackResult = attackRecipe.attack(text, trueLabel)

    print(f"Original Text: {text}")
    print(f"Original Label: {trueLabel}")
    print("Time for Attack:" + attackTimer.timePastAsStr())
    if isinstance(attackResult, SuccessfulAttackResult):
        print("Successful Attack")
        print(f"Attack result: {attackResult}")
        print(f"Perturbed Text: {attackResult.perturbed_result.attacked_text}")

    elif isinstance(attackResult, SkippedAttackResult):
        print("Skipped Attack")
        print(f"Attack result: {attackResult}")
