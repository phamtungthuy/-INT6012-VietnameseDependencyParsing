from utils.constants import TRAIN_FILE_PATH, VALIDATION_FILE_PATH, TEST_FILE_PATH

from corpus.conll_corpus import CONLLCorpus

class ViVTBCorpus(CONLLCorpus):
    def __init__(self):
        self._train = str(TRAIN_FILE_PATH)
        self._dev = str(VALIDATION_FILE_PATH)
        self._test = str(TEST_FILE_PATH)