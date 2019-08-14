import logging
import unittest
import matplotlib.pyplot as plt


class LoggingTest(unittest.TestCase):

    def test_log(self):
        logging.basicConfig(level=logging.DEBUG)
        self._log = logging.getLogger(__name__)

        print("starting test_log")
        self._log.debug("debug output")
        self._log.warning("warning output")
        self._log.error("error output")
        self._log.fatal("fatal output")

    def test_rootlogger(self):
        logging.basicConfig(level=logging.DEBUG)

        print("starting test_rootlogger")

        logging.debug("debug output")
        logging.warning("warning output")
        logging.error("error output")
        logging.fatal("fatal output")

    def test_matplotlib(self):
        f1 = plt.figure("unittest", figsize=(17, 5))
        plt.pause(0.01)
        x, y = f1.get_size_inches()
        assert (x, y) == (17, 5)
        assert not (x, y) == (17, 6)
        print()
