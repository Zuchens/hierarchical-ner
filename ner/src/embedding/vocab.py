from typing import Any


class Vocab:

    @staticmethod
    def add_words_to_vocabulary(vocabulary: dict[str, int], raw_data: list[dict[str, Any]]) -> None:
        for sentence in raw_data:
            for word in sentence['tokens']:
                if word.lower() not in vocabulary:
                    vocabulary[word.lower()] = len(vocabulary)

    def create_vocab(self, train_raw_data: list[dict[str, Any]], test_raw_data: list[dict[str, Any]]) -> dict[str, int]:
        vocabulary = {'PAD': 0, 'UNKNOWN': 1}
        self.add_words_to_vocabulary(vocabulary, train_raw_data)
        self.add_words_to_vocabulary(vocabulary, test_raw_data)
        return vocabulary
