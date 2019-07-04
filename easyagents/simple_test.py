import unittest
import logging

import matplotlib.pyplot as plt
import numpy as np


class LoggingTest(unittest.TestCase):

    def test_log(self):
        logging.basicConfig(level=logging.DEBUG)
        self._log = logging.getLogger(__name__)
        # self._log.setLevel(logging.DEBUG)

        print("starting test_log")
        self._log.debug("debug output")
        self._log.warning("warning output")
        self._log.error("error output")
        self._log.fatal("fatal output")
        return

    def test_rootlogger(self):
        # logging.basicConfig()
        logging.basicConfig(level=logging.DEBUG)

        print("starting test_rootlogger")

        logging.debug("debug output")
        logging.warning("warning output")
        logging.error("error output")
        logging.fatal("fatal output")
        return

    def test_plot(self):
        self.plot_list([1,2,3])

    def plot_list(self, values, xlabel='x', ylabel='y'):
        """ produces a matlib.pyplot plot showing the losses during training.

            Note:
            To see the plot you may call this method from IPython / jupyter notebook.
        """
        value_count = len(values)
        steps = range(0, value_count, 1)
        plt.plot(steps, values)
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)

