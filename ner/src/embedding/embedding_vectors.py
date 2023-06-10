from __future__ import annotations

import numpy as np
from gensim.models import KeyedVectors  # type: ignore[import]
from numpy import ndarray


class EmbeddingVectors:

    @staticmethod
    def create_vectors(model: KeyedVectors, vocabulary: dict[str, int]) -> ndarray:
        vectors = np.zeros((len(vocabulary), model.vector_size))
        word_index = 0
        for index, word in enumerate(vocabulary):
            try:
                vectors[index, :] = model[word]
            except KeyError:    # word not in embedding file
                vectors[index, :] = np.random.rand(model.vector_size)
                word_index += 1
            except AttributeError:
                vectors[index, :] = np.random.rand(model.vector_size)
                word_index += 1
        return vectors
