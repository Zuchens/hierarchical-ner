import re

from ner.src.common.sentence import Word


class NumericalFeatures:

    @staticmethod
    def starts_uppercase(word: str) -> int:
        return 2 if word[0].isupper() else 1

    @staticmethod
    def has_dot(word: str) -> int:
        return 2 if "." in word else 1

    @staticmethod
    def has_num(word: str) -> int:
        return 2 if re.search(r'\d+', word) else 1

    def create_additional_features(self, tokens: list[Word]) -> list[list[int]]:
        return [[
            self.starts_uppercase(word.word),
            self.has_dot(word.word),
            self.has_num(word.word)
        ] for word in tokens]
