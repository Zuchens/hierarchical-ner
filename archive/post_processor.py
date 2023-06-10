from keras.engine.functional import Functional  # type: ignore[import]

from ner.src.common.constants import Constants
from ner.src.common.embedding import Embedding
from ner.src.common.model.model_data import ModelData


class PostProcessor:

    @staticmethod
    def test_validation(
        idx_to_label: dict[int, str],
        val_model_data: ModelData,
        model: Functional,
        embedding: Embedding,
    ) -> None:
        vocabulary = embedding.vocabulary
        idx_to_word = {idx: word for word, idx in vocabulary.items()}
        loss, val_accuracy = model.evaluate([val_model_data.word_features, val_model_data.additional_features],
                                            val_model_data.targets,
                                            verbose=2)
        print(f'Accuracy val (default): %f \n{val_accuracy * 100}')
        val_predictions = model.predict([val_model_data.word_features, val_model_data.additional_features], verbose=2)
        print("Test evaluation")
        # all = 0
        # true = 0
        #
        # predictions = []
        # targets = []
        # sentence_size = val_model_data.word_features.shape[0]
        # labels = {label for label in idx_to_label.values() if label not in {Constants.unknown_word, Constants.pad_word}}
        # for sent_idx in range(sentence_size):
        #     # separate predictions and targets per type
        #     # e.g. {persName: [1,1,0], persName_surName: [0,1,0]}
        #     predictions_sent = dict()
        #     for i in labels:
        #         predictions_sent[i] = [0 for _ in range(ModelParameters.padding)]
        #
        #     targets_sent = dict()
        #     for i in labels:
        #         targets_sent[i] = [0 for _ in range(ModelParameters.padding)]
        #
        #     for token_idx in range(ModelParameters.padding):
        #         idx = np.argmax(val_predictions[sent_idx][token_idx])
        #         idx_target = np.argmax(val_model_data.word_features[sent_idx][token_idx])
        #         if idx_target > 1 or (idx_target == 1 and idx > 2):
        #             if idx_target == idx:
        #                 true += 1
        #             else:
        #                 print("{} : {}\t{}".format(
        #                     idx_to_label[idx], idx_to_label[idx_target],
        #                     idx_to_word[val_model_data.word_features[sent_idx][token_idx]]))
        #             sentence_pred = idx_to_label[idx].split("-")
        #             for i in sentence_pred:
        #                 if (i.startswith("B")):
        #                     predictions_sent[i[2:]][token_idx] = 2
        #                 if (i.startswith("I")):
        #                     predictions_sent[i[2:]][token_idx] = 1
        #
        #             sentence_target = idx_to_label[idx_target].split("-")
        #             for i in sentence_target:
        #                 if (i.startswith("B")):
        #                     targets_sent[i[2:]][token_idx] = 2
        #                 if (i.startswith("I")):
        #                     targets_sent[i[2:]][token_idx] = 1
        #             all += 1
        #     predictions.append(predictions_sent)
        #     targets.append(targets_sent)
        #
        # # save_predictions(offsets, targets, "out/validation_target.json")
        # # save_predictions(offsets, predictions, "out/validation_prediction.json")
        # print('Confusion Matrix')
        # predictions = np.argmax(val_model_data.targets, axis=1).flatten()
        # p = np.argmax(val_predictions, axis=1).flatten()
        #
        # np.savetxt('test.out', confusion_matrix(predictions, p), delimiter=',')
        # acc = true / all if all != 0 else 0
        # print("Accuracy on the validation set " + str(acc))
