from ner.src.common.sentence import InputSentence, Word
from ner.src.preprocess.dependencies_processor import Dependency


class InputSentenceUtils:

    @staticmethod
    def create_based_on_words(raw_words: list[str]) -> InputSentence:
        offset = 0
        words = []
        dependencies = []
        features = []
        for idx, raw_word in enumerate(raw_words):
            words.append(Word(word=raw_word, word_offset=offset, word_embedding_idx=idx))
            dependencies.append(Dependency(str(idx), "root", 0))
            features.append([0, 0, 0])
            offset += offset
        return InputSentence(dependencies=dependencies, words=words, features=features)
