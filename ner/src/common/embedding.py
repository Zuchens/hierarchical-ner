from dataclasses import dataclass

from numpy import ndarray


@dataclass
class Embedding:
    vocabulary: dict
    vectors: ndarray
