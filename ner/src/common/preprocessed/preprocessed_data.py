from dataclasses import dataclass

from ner.src.common.preprocessed.processed_document import ProcessedDocument
from ner.src.common.sentence import Sentence


@dataclass
class PreprocessedData:
    label_to_idx: dict[str, int]
    dependency_label_to_idx: dict[str, int]
    processed_documents: list[ProcessedDocument]

    @property
    def processed_sentences(self) -> list[Sentence]:
        return [sentence for document in self.processed_documents for sentence in document.sentences]
