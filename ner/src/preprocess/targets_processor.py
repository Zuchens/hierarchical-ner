from ner.src.common.sentence import InputSentence
from ner.src.preprocess.outputs import OutputProcessor, RawTargetType

ProcessedTargetType = list[int]
ProcessedTargets = list[ProcessedTargetType]


class TargetsProcessor:

    def __init__(self, label_to_idx=None):
        self.output_processor = OutputProcessor(label_to_idx)

    def get_targets_by_sentences(self, input_documents: list[list[InputSentence]],
                                 entities: list[RawTargetType]) -> list[ProcessedTargets]:
        output_indexes_by_document = self.output_processor.get_targets_idx(entities)
        document_targets = []
        for document, output_per_document in zip(input_documents, output_indexes_by_document):
            document_targets.append(self.align_to_sentence(document, output_per_document))
        return document_targets

    def align_to_sentence(
        self,
        input_sentences: list[InputSentence],
        output_indexes_by_document: list[int],
    ) -> ProcessedTargets:
        target_sentences = []
        target_word_index = 0
        for sentence in input_sentences:
            sentence_target = []
            for _ in sentence.words:
                document_target = output_indexes_by_document[target_word_index]
                sentence_target.append(document_target)
                target_word_index += 1
            target_sentences.append(sentence_target)
        assert [len(sent.words) for sent in input_sentences] == [len(sent) for sent in target_sentences]
        return target_sentences
