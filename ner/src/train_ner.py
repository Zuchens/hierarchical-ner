#!/usr/bin/env python
import argparse
import json

from ner.src.embedding.embedding_creator import EmbeddingCreator
from ner.src.preprocess.preprocess import Preprocessor
from ner.src.train.train import Trainer


def main(parsed_arguments: argparse.Namespace) -> None:
    with open(parsed_arguments.train, encoding="utf8") as file:
        train_raw_data = json.load(file)["texts"]
    with open(parsed_arguments.test, encoding="utf8") as file:
        test_raw_data = json.load(file)["texts"]
    embeddings = EmbeddingCreator().load_embeddings(
        train_raw_data=train_raw_data,
        test_raw_data=test_raw_data,
    )
    preprocessed_data = Preprocessor().preprocess_training_data(
        embeddings=embeddings,
        train_raw_data=train_raw_data,
    )
    Trainer().train_and_eval(embeddings, preprocessed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER model training')
    parser.add_argument('--train',
                        type=str,
                        help='train dataset file path',
                        default="data/input_data/train/train-sample.json")
    parser.add_argument('--test', type=str, help='test dataset file path', default="data/test_data/test-sample.json")
    arguments = parser.parse_args()
    main(arguments)
