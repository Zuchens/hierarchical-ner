from keras.callbacks import History    # type: ignore[import]
from matplotlib import pyplot as plt    # type: ignore[import]


class TrainVisualization:

    @staticmethod
    def plot_history(history: History) -> None:
        print(history.history.keys())
        TrainVisualization.plot_accuracy(history)
        TrainVisualization.plot_loss(history)

    @staticmethod
    def plot_loss(history: History) -> None:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')

    @staticmethod
    def plot_accuracy(history: History) -> None:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.clf()
