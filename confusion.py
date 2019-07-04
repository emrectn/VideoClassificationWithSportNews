from data import DataSet
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

data = DataSet(
    seq_length=36,
    class_limit=None
)

val_generator = data.frame_generator(355, 'test', "features")
a = 0
X = []
y = []

for i, j in val_generator:
    a = a+1
    X.append(i)
    y.append(j)
    if a == 1:
        break

model = load_model("data/checkpoints/lstm-features.016-0.125.hdf5")

predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)

true_classes = np.argmax(y[0], axis=1)
print(true_classes)

cm = confusion_matrix(true_classes, predicted_classes)
cm_plot_labels = ['Archery', 'BalanceBeam', 'BaseballPitch', 'Basketball', 'BoxingPunchingBag', 'BreastStroke', 'GolfSwing',
                  'HammerThrow', 'HighJump', 'HorseRiding', 'Kayaking', 'SkyDiving', 'SoccerPenalty', 'Surfing']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
