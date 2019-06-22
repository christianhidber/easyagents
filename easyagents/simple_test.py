import unittest
import logging

class LoggingTest(unittest.TestCase):

    def setUp(self):
        pass

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

