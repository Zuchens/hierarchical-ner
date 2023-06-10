from dataclasses import dataclass

from numpy import ndarray


@dataclass
class ModelData:
    additional_features: ndarray
    word_features: ndarray
    targets: ndarray
