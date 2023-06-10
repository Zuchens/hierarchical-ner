from dataclasses import dataclass

from numpy import ndarray
from numpy._typing import ArrayLike


@dataclass
class Embedding:
    vocabulary: dict
    vectors: ndarray