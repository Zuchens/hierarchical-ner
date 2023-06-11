from __future__ import annotations

from dataclasses import dataclass

from ner.src.common.constants import Constants
from ner.src.common.embedding import Embedding
from ner.src.common.preprocessed.raw_document import RawDocument
from ner.src.common.sentence import Sentence, Word, InputSentence
from ner.src.preprocess.dependencies_processor import DependenciesProcessor
from ner.src.preprocess.numerical_features import NumericalFeatures


@dataclass
class WordIndex:
    index: int = 0


@dataclass
class ProcessedDocument:
    sentences: list[Sentence]


class InputProcessor:

    def get_sentences(
        self,
        raw_document: RawDocument,
        dependencies_processor: DependenciesProcessor,
        embeddings: Embedding,
    ) -> list[InputSentence]:
        # the sentence splitting was done during external model in dependency parsing
        # therefore we align sentences to match split in dependencies
        word_index = WordIndex()
        sentences = []
        for sentence_dependencies, sentence_dependency_labels in raw_document.dependencies_with_labels:
            sentence_size = len(sentence_dependencies)
            dependencies = dependencies_processor.set_dependencies(sentence_dependencies, sentence_dependency_labels)
            words = self.align_words_to_sentences(raw_document, sentence_size, word_index, embeddings.vocabulary)
            features = NumericalFeatures().create_additional_features(words)
            assert len(raw_document.entities) == len(raw_document.tokens), "Labels do not match tokens"
            sentence = InputSentence(dependencies=dependencies, words=words, features=features)
            sentences.append(sentence)
        return sentences

    def align_words_to_sentences(self, doc: RawDocument, sentence_size: int, word_idx: WordIndex,
                                 word_to_index: dict[str, int]) -> list[Word]:
        words = []
        for _ in range(sentence_size):
            word = doc.tokens[word_idx.index].lower()
            embedding_idx = word_to_index.get(word, word_to_index[Constants.unknown_word])
            words.append(Word(word=word, word_offset=doc.offsets[word_idx.index], word_embedding_idx=embedding_idx))
            word_idx.index += 1
        return words
