from tensorflow.keras import Input, Model   # type: ignore[import]
from tensorflow.keras.layers import Embedding, Concatenate, Masking, Dropout, Bidirectional, TimeDistributed, Dense, GRU # type: ignore[import]
from keras.utils import plot_model # type: ignore[import]
from ner.src.common.constants import ModelParameters


class NERClassifier:

    def create_model(self, embeddings, emb_features, feature_size, output_size):
        features = Input(shape=(ModelParameters.padding, feature_size,), name='words')
        embedding_input = Input(shape=(ModelParameters.padding,), name='embedding_input')
        embedding = Embedding(
            embeddings.shape[0],
            emb_features,
            input_length=ModelParameters.padding,
            weights=[embeddings],
            trainable=ModelParameters.trainable_embeddings)(embedding_input)
        concatenated_input = Concatenate(axis=-1)([embedding, features])
        mask_in = Masking()(concatenated_input)
        rnn = Bidirectional(
            GRU(ModelParameters.output_dim_rnn,
                activation=ModelParameters.activation_rnn,
                return_sequences=True))(mask_in)
        dropout = Dropout(ModelParameters.dropout)(rnn)
        time_distributed_dense = TimeDistributed(
            Dense(ModelParameters.dense_size, activation=ModelParameters.activation_dense)
        )(dropout)
        mask_out = Masking()(time_distributed_dense)
        out = Dense(output_size, activation=ModelParameters.activation_output)(mask_out)
        model = Model(inputs=[embedding_input, features], outputs=[out])
        model.compile(optimizer=ModelParameters.optimizer,
                      loss=ModelParameters.loss_function,
                      metrics=['accuracy'])
        model.summary()

        plot_model(model, to_file='model.png', show_shapes=True)
        return model
