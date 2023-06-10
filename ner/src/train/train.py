from keras.engine.functional import Functional # type: ignore[import]
from sklearn.model_selection import train_test_split # type: ignore[import]

from ner.src.common.constants import ModelParameters
from ner.src.common.embedding import Embedding
from ner.src.common.model.model_data import ModelData
from ner.src.postprocessing.post_processor import PostProcessor
from ner.src.preprocess.preprocess import PreprocessedData
from ner.src.train.model_feature_processor import TrainFeaturesProcessor
from ner.src.train.models.flat_rnn_model import NERClassifier
from ner.src.train.visualization import Visualization


class Trainer:

    def __init__(self):
        self.train_processor = TrainFeaturesProcessor()

    def train_and_eval(self, vocab: Embedding,
                       preprocessed_data: PreprocessedData) -> None:
        data_train, data_val = train_test_split(
            preprocessed_data.processed_sentences,
            test_size=ModelParameters.validation_size,
            shuffle=True)
        train_model_data = self.train_processor.prepare_data(data_train, preprocessed_data, vocab)
        model = self.train(train_model_data, vocab)
        val_model_data = self.train_processor.prepare_data(data_val, preprocessed_data, vocab)

        idx_to_label = {idx: label for label, idx in preprocessed_data.label_to_idx.items()}
        PostProcessor.test_validation(idx_to_label, val_model_data, model, vocab)

    def train(self, train_model_data: ModelData, vocab: Embedding) -> Functional:
        vectors = vocab.vectors
        model = NERClassifier().create_model(vectors,
                                             emb_features=vectors.shape[1],
                                             feature_size=train_model_data.additional_features.shape[2],
                                             output_size=len(train_model_data.targets[0][0]))
        history = model.fit([train_model_data.word_features, train_model_data.additional_features],
                            train_model_data.targets,
                            batch_size=ModelParameters.batch_size,
                            epochs=ModelParameters.epochs,
                            validation_split=ModelParameters.validation_size,
                            verbose=2)

        Visualization.plot_history(history)
        return model
