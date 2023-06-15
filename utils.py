__author__ = "Moaad Ben Amar"

from datetime import datetime
import time
import logging


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
