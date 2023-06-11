from typing import Any

from ner.src.common.constants import Constants


class Vocab:

    @staticmethod
    def add_words_to_vocabulary(vocabulary: dict[str, int], raw_data: list[dict[str, Any]]) -> None:
        for sentence in raw_data:
            for word in sentence['tokens']:
                if word.lower() not in vocabulary:
                    vocabulary[word.lower()] = len(vocabulary)

    def create_vocab(self, train_raw_data: list[dict[str, Any]], test_raw_data: list[dict[str, Any]]) -> dict[str, int]:
        vocabulary = {Constants.pad_word: 0, Constants.unknown_word: 1}
        self.add_words_to_vocabulary(vocabulary, train_raw_data)
        self.add_words_to_vocabulary(vocabulary, test_raw_data)
        return vocabulary
