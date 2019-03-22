from sklearn.metrics import confusion_matrix
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='test_result.json')
parser.add_argument('--title', type=str, default='title')

args = parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():

    print(args.input_file)
    file = args.input_file.split('/')[1]
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    with open(args.input_file) as f:
        data = json.load(f)

    y_true, y_pred = data['list_of_gt'], data['list_of_pred']

    class_names = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11']
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=class_names, normalize=False, title=args.title)
    plt.savefig('./plots/conf_matrix_{}.jpg'.format(file[:-5]))

    class_correct = data['class_correct']
    class_total = data['class_total']

    print('\n')
    print('overall accuracy: {}\n'.format(data['accuracy']))
    accuracy = 0.
    for i in range(len(class_names)):
       print('Accuracy of {} : {:.3f}%'.format(
            class_names[i], 100 * class_correct[i] / class_total[i]))





if __name__ == '__main__':
    main()