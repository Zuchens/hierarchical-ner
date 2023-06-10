from __future__ import annotations

import os
from typing import Any

from gensim.models import KeyedVectors  # type: ignore[import]

from ner.src.common.constants import Constants
from ner.src.common.embedding import Embedding
from ner.src.embedding.embedding_vectors import EmbeddingVectors
from ner.src.embedding.vocab import Vocab


class InvalidModelPathException(Exception):
    pass


class EmbeddingCreator:

    def load_embeddings(self, train_raw_data: list[dict[str, Any]], test_raw_data: list[dict[str, Any]]) -> Embedding:
        # Creating vocab from train and test file
        vocabulary = Vocab().create_vocab(train_raw_data, test_raw_data)
        model = self.load_embedding_file(Constants.emb_path)
        vectors = EmbeddingVectors().create_vectors(model, vocabulary)
        return Embedding(vocabulary, vectors)

    @staticmethod
    def load_embedding_file(embeddings_path: str) -> KeyedVectors:
        if os.path.isfile(f"{embeddings_path}.model"):
            model = KeyedVectors.load(f"{embeddings_path}.model")
        elif os.path.isfile(f"{embeddings_path}.vec"):
            model = KeyedVectors.load_word2vec_format(f"{embeddings_path}.vec")
        else:
            raise InvalidModelPathException("No valid path to embeddings")
        return model
