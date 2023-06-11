from ner.src.common.embedding import Embedding
from ner.src.common.preprocessed.preprocessed_data import PreprocessedData
from ner.src.common.preprocessed.processed_document import InputProcessor, ProcessedDocument
from ner.src.common.preprocessed.raw_document import RawDocument
from ner.src.common.sentence import Sentence, InputSentence

from ner.src.preprocess.dependencies_processor import DependenciesProcessor
from ner.src.preprocess.targets_processor import TargetsProcessor, ProcessedTargets

InputDocumentType = list[InputSentence]


class Preprocessor:

    def __init__(self):
        self.dependencies_processor = DependenciesProcessor()
        self.targets_processor = TargetsProcessor()

    def preprocess_training_data(self, embeddings: Embedding, train_raw_data: list) -> PreprocessedData:
        filtered_raw_data = [doc for doc in train_raw_data if doc["tokens"]]
        processed_inputs, raw_documents = self.create_raw_documents_and_inputs(embeddings, filtered_raw_data)
        entities = [doc.entities for doc in raw_documents]
        document_targets = self.targets_processor.get_targets_by_sentences(input_documents=processed_inputs,
                                                                           entities=entities)
        processed_documents = self.create_processed_documents(document_targets, processed_inputs)
        return PreprocessedData(label_to_idx=self.targets_processor.output_processor.labels,
                                dependency_label_to_idx=self.dependencies_processor.dependency_label_to_idx,
                                processed_documents=processed_documents)

    def create_raw_documents_and_inputs(
        self,
        embeddings: Embedding,
        filtered_raw_data: list,
    ) -> tuple[list[InputDocumentType], list[RawDocument]]:
        raw_documents = []
        processed_inputs = []
        for document_content in filtered_raw_data:
            raw_document = RawDocument(**document_content)
            raw_documents.append(raw_document)
            processed_input = InputProcessor().get_sentences(raw_document=raw_document,
                                                             dependencies_processor=self.dependencies_processor,
                                                             embeddings=embeddings)

            processed_inputs.append(processed_input)
        return processed_inputs, raw_documents

    def create_processed_documents(self, document_targets: list[ProcessedTargets],
                                   processed_inputs: list[InputDocumentType]) -> list[ProcessedDocument]:
        processed_documents = []
        for processed_input, target in zip(processed_inputs, document_targets):
            sentences = []
            for input_sentence, target_sentence in zip(processed_input, target):
                sentences.append(Sentence(input_sentence, target_sentence))
            processed_documents.append(ProcessedDocument(sentences))
        return processed_documents
