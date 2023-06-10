from dataclasses import dataclass
from typing import Optional

from ner.src.common.preprocessed.processed_document import ProcessedDocument
from ner.src.common.preprocessed.raw_document import RawDocument


@dataclass
class Document:
    raw_document: RawDocument
    processed_document: Optional[ProcessedDocument] = None
