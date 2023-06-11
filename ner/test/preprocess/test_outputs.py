from unittest import TestCase

from ner.src.common.constants import Constants
from ner.src.preprocess.outputs import OutputProcessor, RawTargets


class OutputProcessorTestCase(TestCase):

    def test_convert_concatenated_labels_to_indices(self) -> None:
        # GIVEN
        targets = [[{Constants.outside_label}, {"person", "person_firstname"}, {"person"}],
                   [{Constants.outside_label}, {Constants.outside_label}, {"person_firstname", "person"}]]
        expected_output = [[1, 2, 3], [1, 1, 2]]

        output_processor = OutputProcessor(None)
        expected_output_processor_labels = {
            Constants.pad_label: 0,
            Constants.outside_label: 1,
            "person-person_firstname": 2,
            "person": 3
        }

        # WHEN
        result = output_processor.convert_concatenated_labels_to_indices(targets)

        # THEN
        self.assertCountEqual(expected_output, result)
        self.assertCountEqual(expected_output_processor_labels, output_processor.labels)

    def test_get_sentence_target_labels(self) -> None:
        # GIVEN
        raw_sentence_targets: RawTargets = [[],
                                            [{
                                                "type": "person",
                                                "offsets": [{
                                                    "from": 1,
                                                    "to": 3
                                                }, {
                                                    "from": 2,
                                                    "to": 3
                                                }]
                                            }, {
                                                "type": "person",
                                                "subtype": "firstname",
                                                "offsets": [{
                                                    "from": 1,
                                                    "to": 3
                                                }]
                                            }], []]
        expected_output = [{Constants.outside_label}, {"person", "person_firstname"}, {"person"}]

        # WHEN
        result = OutputProcessor(None).get_sentence_target_labels(raw_sentence_targets)

        # THEN
        self.assertCountEqual(result, expected_output)
