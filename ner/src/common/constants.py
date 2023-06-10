class Constants:
    unknown_word: str = "UNKNOWN"
    pad_word: str = "PAD"
    outside_label: str = "Outside"
    pad_label: str = "P"
    use_test_file = False
    emb_path: str = "data/embedding/small/small"


class ModelParameters:
    padding: int = 300
    output_dim_rnn: int = 300
    activation_rnn: str = "relu"
    dropout: float = 0.6
    trainable_embeddings: bool = True
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    activation_output: str = "softmax"
    activation_dense: str = "relu"
    dense_size: int = 600
    epochs: int = 1
    validation_size: float = 0.1
    batch_size: int = 128
