import argparse
import json

from ner.src.embedding.embedding_creator import EmbeddingCreator
from ner.src.preprocess.preprocess import Preprocessor
from ner.src.train.train import Trainer


def main(arguments: argparse.Namespace) -> None:
    with open(arguments.train, encoding="utf8") as f:
        train_raw_data = json.load(f)["texts"]
    with open(arguments.test, encoding="utf8") as f:
        test_raw_data = json.load(f)["texts"]
    embeddings = EmbeddingCreator().load_embeddings(train_raw_data=train_raw_data,
                                                    test_raw_data=test_raw_data, )
    preprocessed_data = Preprocessor().preprocess_training_data(
        embeddings=embeddings,
        train_raw_data=train_raw_data, )
    Trainer().train_and_eval(embeddings, preprocessed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER model training')
    parser.add_argument('--train', type=str, help='train dataset file path',
                        default="data/input_data/train/out-small-2.json")
    parser.add_argument('--test', type=str, help='test dataset file path',
                        default="data/test_data/results.json")
    parser.add_argument('--out', type=str, help='train output file path',
                        default="data/output_data/train_file.txt")
    arguments = parser.parse_args()
    main(arguments)
