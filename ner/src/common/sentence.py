from dataclasses import dataclass
from typing import Any, Optional

from ner.src.preprocess.dependencies_processor import Dependency


@dataclass
class Word:
    word_offset: int
    word: str
    word_embedding_idx: Optional[int] = None


@dataclass
class Sentence:
    dependencies: list[Dependency]
    words: list[Word]
    features: Optional[list[list[int]]] = None
    targets: Optional[list[int]] = None
    offsets: Optional[list[int]] = None

    def __repr__(self) -> str:
        return " ".join([word.word for word in self.words])

    def __str__(self) -> str:
        return " ".join([word.word for word in self.words])

    def get_padded_words_idx(self, padding_size: int, padding_idx: int) -> list[int]:
        words_idx = [word.word_embedding_idx for word in self.words][:padding_size]
        words_idx += [padding_idx for _ in range(padding_size - len(self.words))]
        return words_idx

    def get_padded_dependency_labels_idx(self, padding_size: int) -> list[int]:
        dependencies_labels_idx = [dependency.dependency_label_idx for dependency in self.dependencies][:padding_size]
        dependencies_labels_idx += [0 for _ in range(padding_size - len(self.dependencies))]
        return dependencies_labels_idx

    def get_padded_features(self,  padding_size: int) -> list[list[int]]:
        features_number = len(self.features[0])
        features_idx = self.features[:padding_size]
        features_idx += [[0 for _ in range(features_number)] for _ in range(padding_size - len(self.features))]
        return features_idx

    def get_padded_dependencies(self,  padding_size: int) -> list[int]:
        dependencies_idx = [int(dependency.dependency_idx) for dependency in self.dependencies][:padding_size]
        dependencies_idx += [0 for _ in range(padding_size - len(self.dependencies))]
        return dependencies_idx

    def get_padded_target(self, padding_size: int) -> list[int]:
        targets = self.targets[:padding_size]
        targets += [0 for _ in range(padding_size - len(self.targets))]
        return targets
