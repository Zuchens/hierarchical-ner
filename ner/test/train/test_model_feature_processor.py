from unittest import TestCase

import numpy

from ner.src.train.model_feature_processor import TrainFeaturesProcessor


class TrainFeaturesProcessorTestCase(TestCase):

    def test_one_hot_encode(self):
        # GIVEN
        labels = [1, 2, 0]
        label_to_idx = {'person': 0, 'person_surname': 1, 'geo': 2}

        expected_output = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

        # WHEN
        results = TrainFeaturesProcessor().one_hot_encode(labels, label_to_idx)

        # THEN
        numpy.alltrue(results == expected_output)
