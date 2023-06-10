import numpy as np
from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import]

from ner.src.common.constants import ModelParameters, Constants
from ner.src.common.embedding import Embedding
from ner.src.common.model.model_data import ModelData
from ner.src.common.preprocessed.preprocessed_data import PreprocessedData
from ner.src.common.sentence import Sentence


class TrainFeaturesProcessor:

    def get_additional_features(self, sentences: list[Sentence], dep2idx: dict[str, int]) -> ndarray:
        padding = ModelParameters.padding
        numerical_features = np.array([sentence.input.get_padded_features(padding) for sentence in sentences])
        dependency_idx = np.reshape(
            np.array([sentence.input.get_padded_dependencies(padding) for sentence in sentences]),
            (numerical_features.shape[0], numerical_features.shape[1], 1))
        dependency_labels = np.array([
            self.one_hot_encode(sentence.input.get_padded_dependency_labels_idx(padding), dep2idx) for sentence in
            sentences
        ])
        features = np.concatenate((numerical_features, dependency_idx, dependency_labels), axis=2)
        return features

    def prepare_data(self, data_train: list[Sentence], preprocessed_data: PreprocessedData,
                     vocab: Embedding) -> ModelData:
        numerical_features = self.get_additional_features(data_train, preprocessed_data.dependency_label_to_idx)
        targets = self.get_targets(preprocessed_data.label_to_idx, data_train)
        word_features = self.get_input(data_train, vocab)
        return ModelData(additional_features=numerical_features, word_features=word_features, targets=targets)

    def get_input(self, sentences: list[Sentence], vocab: Embedding) -> ndarray:
        padding = ModelParameters.padding
        return np.array([
            sentence.input.get_padded_words_idx(padding, vocab.vocabulary[Constants.pad_word])
            for sentence in sentences
        ])

    def get_targets(self, label_to_idx: dict[str, int], sentences: list[Sentence]) -> ndarray:
        padding = ModelParameters.padding
        target = [self.one_hot_encode(sentence.get_padded_target(padding), label_to_idx) for sentence in sentences]
        return np.array(target)

    def one_hot_encode(self, categories: list[int], label_to_idx: dict[str, int]) -> ndarray:
        one_hot_categories = OneHotEncoder(sparse_output=False)
        one_hot_categories.fit([[x] for x in label_to_idx.values()])
        return one_hot_categories.transform([[i] for i in categories])
