from typing import List

from ner.src.common.preprocessed.document import Document
from ner.src.preprocess.outputs import OutputProcessor


class TargetsProcessor:
    def __init__(self, label2idx=None):
        self.output_processor = OutputProcessor(label2idx)

    def get_targets_by_sentences(self, documents: List[Document]) -> dict[str, int]:
        output_indexes_by_document = self.output_processor.get_targets_idx(documents)

        for document, output_per_document in zip(documents, output_indexes_by_document):
            self.align_to_sentence(document, output_per_document)
        return self.output_processor.labels

    def align_to_sentence(self, document: Document,
                          output_indexes_by_document: list[int],) -> None:
        sentences = document.processed_document.sentences
        target_word_index = 0
        for sentence in sentences:
            sentence_target = []
            for _ in sentence.words:
                document_target = output_indexes_by_document[target_word_index]
                sentence_target.append(document_target)
                target_word_index += 1
            sentence.targets = sentence_target
        assert [len(sentence.words) for sentence in sentences] == [len(sentence.targets) for sentence in sentences]
