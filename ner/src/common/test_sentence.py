from unittest import TestCase

from ner.src.common.sentence import Word, InputSentence
from ner.src.preprocess.dependencies_processor import Dependency


class InputSentenceTestCase(TestCase):

    def setUp(self):
        words = [
            Word(word="Hello", word_offset=0, word_embedding_idx=1),
            Word(word="world", word_offset=6, word_embedding_idx=2)
        ]
        dependencies = [
            Dependency(dependency_label="root", dependency_label_idx=0, dependency_idx="-1"),
            Dependency(dependency_label="intj", dependency_label_idx=1, dependency_idx="0")
        ]
        features = [[0, 1, 0], [1, 1, 0]]
        self.input_sentence = InputSentence(words=words, dependencies=dependencies, features=features)

    def test_get_padded_words_idx(self):
        # GIVEN
        expected_result = [1, 2, 0, 0]
        # WHEN
        result = self.input_sentence.get_padded_words_idx(4, 0)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_words_idx_shorter(self):
        # GIVEN
        expected_result = [1]
        # WHEN
        result = self.input_sentence.get_padded_words_idx(1, 0)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_dependency_labels_idx(self):
        # GIVEN
        expected_result = [0, 1, 0, 0]
        # WHEN
        result = self.input_sentence.get_padded_dependency_labels_idx(4)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_dependency_labels_idx_shorter(self):
        # GIVEN
        expected_result = [0]
        # WHEN
        result = self.input_sentence.get_padded_dependency_labels_idx(1)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_features(self):
        # GIVEN
        expected_result = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]
        # WHEN
        result = self.input_sentence.get_padded_features(4)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_features_shorter(self):
        # GIVEN
        expected_result = [[0, 1, 0]]
        # WHEN
        result = self.input_sentence.get_padded_features(1)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_dependencies(self):
        # GIVEN
        expected_result = [-1, 0, 0, 0]
        # WHEN
        result = self.input_sentence.get_padded_dependencies(4)
        # THEN
        self.assertListEqual(expected_result, result)

    def test_get_padded_dependencies_shorter(self):
        # GIVEN
        expected_result = [-1]
        # WHEN
        result = self.input_sentence.get_padded_dependencies(1)
        # THEN
        self.assertListEqual(expected_result, result)
