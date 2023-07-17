__author__ = "Moaad Ben Amar"

from datetime import datetime
import time
import logging


def loadTxtAsLst(fileName):
    filePath = f"Injection_Prompts/{fileName}.txt"
    with open(filePath, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


class TextAttackLogger:
    def __init__(self, log_file="LOGS/AdversarialAttack"):
        self.logger = None  # tflogging.get_logger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        # Create a StreamHandler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Create a FileHandler to write logs to a file
        # file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(console_handler)
        # self.logger.addHandler(file_handler)

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


class Logger:
    def __init__(self):
        self._logger = logging.getLogger("AdversarialAttack")
        self._logger.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter("%(process)d: %(asctime)s-%(levelname)s---%(message)s",
                                      datefmt='%d.%b.%y %H:%M:%S')

        # FileHandler
        currTime = datetime.now()
        formattedTime = currTime.strftime("%d.%m.%y %H-%M")
        fileHandler = logging.FileHandler(f'LOGS/AdversarialAttack {formattedTime}.log')
        fileHandler.setFormatter(formatter)
        self._logger.addHandler(fileHandler)

        # StreamHandler
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg):
        self._logger.error(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def log(self, msg):
        self._logger.info(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def debug(self, msg):
        self._logger.debug(msg)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def timePast(self, returnFormat="sec"):
        """
        Calculate the elapsed time since `self.start_time` based on the provided `returnFormat` parameter.

        Example usage:
            instance = ClassName()
            instance.start_time = time.time()  # Set the start time
            elapsed_time = instance.timePast(returnFormat="min")  # Calculate elapsed time in minutes

        :param returnFormat: Specifies the desired format for the time value. Valid options are "sec" (seconds),
                                    "min" (minutes), or "hrs" (hours). Defaults to "sec".
        :return: float: The elapsed time since `self.start_time` in the specified format.
        """
        formatter = 1
        if returnFormat == "min":
            formatter = 60
        elif returnFormat == "hrs":
            formatter = 3600
        return (time.time() - self.start_time) / formatter

    def timePastAsStr(self, returnFormat="sec"):
        return str(self.timePast(returnFormat))
