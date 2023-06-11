from unittest import TestCase
from unittest.mock import Mock

from ner.src.preprocess.targets_processor import TargetsProcessor
from ner.test.utils import InputSentenceUtils


class TargetProcessorTestCase(TestCase):

    def test_get_targets_by_sentences(self):
        # GIVEN
        input_documents = [[
            InputSentenceUtils.create_based_on_words(["Hello"]),
            InputSentenceUtils.create_based_on_words(["Anna"])
        ], [InputSentenceUtils.create_based_on_words(["Lorem", "ipsum"])]]

        first_sentence_entities = [[]]
        second_sentence_entities = [[{"type": "persName"}]]
        third_sentence_entities = [[], []]
        entities = [[first_sentence_entities, second_sentence_entities], [third_sentence_entities]]

        targets_processor = TargetsProcessor()
        targets_processor.output_processor = Mock()
        targets_processor.output_processor.get_targets_idx.return_value = [[1, 2], [1, 1]]

        expected_docments = [[[1], [2]], [[1, 1]]]
        # WHEN
        result = targets_processor.get_targets_by_sentences(input_documents, entities)
        # THEN
        self.assertListEqual(expected_docments, result)

    def test_align_to_sentence(self):
        # GIVEN
        input_sentences = [
            InputSentenceUtils.create_based_on_words(["Hello", "world"]),
            InputSentenceUtils.create_based_on_words(["This", "is", "me"])
        ]
        output_indexes_by_document = [1, 2, 1, 0, 1]
        expected_targets = [[1, 2], [1, 0, 1]]

        # WHEN
        results = TargetsProcessor().align_to_sentence(input_sentences=input_sentences,
                                                       output_indexes_by_document=output_indexes_by_document)

        # THEN
        self.assertListEqual(expected_targets, results)
