from ner.src.common.embedding import Embedding
from ner.src.common.preprocessed.preprocessed_data import PreprocessedData
from ner.src.common.preprocessed.processed_document import InputProcessor, ProcessedDocument
from ner.src.common.preprocessed.raw_document import RawDocument
from ner.src.common.sentence import Sentence
from ner.src.preprocess.dependencies_processor import DependenciesProcessor
from ner.src.preprocess.targets_processor import TargetsProcessor


class Preprocessor:

    def __init__(self):
        self.dependencies_processor = DependenciesProcessor()
        self.targets_processor = TargetsProcessor()

    def preprocess_training_data(self, embeddings: Embedding, train_raw_data: dict) -> PreprocessedData:
        filtered_raw_data = [doc for doc in train_raw_data if doc["tokens"]]
        raw_documents = []
        processed_inputs = []
        for document_content in filtered_raw_data:
            raw_document = RawDocument(**document_content)
            raw_documents.append(raw_document)
            processed_input = InputProcessor.get_sentences(raw_document, self.dependencies_processor, embeddings)
            processed_inputs.append(processed_input)
        document_targets = self.targets_processor.get_targets_by_sentences(input_documents=processed_inputs,
                                                                           entities=[doc.entities for doc in
                                                                                     raw_documents])
        processed_documents = []
        for processed_input, target in zip(processed_inputs, document_targets):
            sentences = []
            for input_sentence, target_sentence in zip(processed_input, target):
                sentences.append(Sentence(input_sentence, target_sentence))
            processed_documents.append(ProcessedDocument(sentences))
        return PreprocessedData(label_to_idx=self.targets_processor.output_processor.labels,
                                dependency_label_to_idx=self.dependencies_processor.dependency_label2idx,
                                processed_documents=processed_documents)
