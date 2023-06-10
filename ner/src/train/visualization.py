from matplotlib import pyplot as plt


class Visualization:

    @staticmethod
    def plot_history(history):
        print(history.history.keys())
        Visualization.plot_accuracy(history)
        Visualization.plot_loss(history)

    @staticmethod
    def plot_loss(history) -> None:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')

    @staticmethod
    def plot_accuracy(history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.clf()
