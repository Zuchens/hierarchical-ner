from keras.engine.functional import Functional    # type: ignore[import]

from ner.src.common.model.model_data import ModelData


class PostProcessor:

    @staticmethod
    def test_validation(
        val_model_data: ModelData,
        model: Functional,
    ) -> None:
        loss, val_accuracy = model.evaluate([val_model_data.word_features, val_model_data.additional_features],
                                            val_model_data.targets,
                                            verbose=2)
        print(f'Accuracy loss {loss} val (default):\n{val_accuracy * 100}')
