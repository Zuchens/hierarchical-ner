from unittest import TestCase
from unittest.mock import Mock

from ner.src.common.sentence import Word
from ner.src.preprocess.numerical_features import NumericalFeatures


class NumericalFeaturesTestCase(TestCase):

    def test_starts_uppercase_true(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.starts_uppercase("Uppercase"), 2)

    def test_starts_uppercase_false(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.starts_uppercase("notUppercase"), 1)

    def test_has_dot_true(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.has_dot("with.dot"), 2)

    def test_has_dot_false(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.has_dot("without_dot"), 1)

    def test_has_num_true(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.has_num("has_1_number"), 2)

    def test_has_num_false(self):
        # GIVEN-WHEN-THEN
        self.assertEqual(NumericalFeatures.has_num("no_number"), 1)

    def test_create_additional_features(self):
        # GIVEN
        numerical_features = NumericalFeatures()
        numerical_features.starts_uppercase = Mock()
        numerical_features.starts_uppercase.return_value = 1
        numerical_features.has_dot = Mock()
        numerical_features.has_dot.return_value = 2
        numerical_features.has_num = Mock()
        numerical_features.has_num.return_value = 3

        tokens = [Word(word="first", word_offset=0,word_embedding_idx=1),
                  Word(word="word", word_offset=7,word_embedding_idx=2)]
        expected_output = [[1, 2, 3], [1, 2, 3]]
        # WHEN
        result = numerical_features.create_additional_features(tokens)

        # THEN
        self.assertEqual(result, expected_output)
