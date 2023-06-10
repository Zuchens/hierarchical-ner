from ner.src.common.embedding import Embedding
from ner.src.common.preprocessed.document import Document
from ner.src.common.preprocessed.preprocessed_data import PreprocessedData
from ner.src.common.preprocessed.processed_document import ProcessedDocument
from ner.src.common.preprocessed.raw_document import RawDocument
from ner.src.preprocess.dependencies_processor import DependenciesProcessor
from ner.src.preprocess.targets_processor import TargetsProcessor


class Preprocessor:

    def __init__(self):
        self.dependencies_processor = DependenciesProcessor()
        self.targets_processor = TargetsProcessor()

    def preprocess_training_data(self, embeddings: Embedding, train_raw_data: dict) -> PreprocessedData:
        filtered_raw_data = [doc for doc in train_raw_data if doc["tokens"]]
        documents = []
        for document_content in filtered_raw_data:
            documents.append(Document(raw_document=RawDocument(**document_content)))
        self.add_preprocessed_document(embeddings, documents)
        label2idx = self.targets_processor.get_targets_by_sentences(documents)
        return PreprocessedData(label_to_idx=label2idx,
                                dependency_label_to_idx=self.dependencies_processor.dependency_label2idx,
                                processed_documents=[document.processed_document for document in documents])

    def add_preprocessed_document(self, embeddings, documents: list[Document]) -> None:
        for doc in documents:
            doc.processed_document = ProcessedDocument.create(doc.raw_document, self.dependencies_processor, embeddings)