import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from pickle import load
import torch
import misc.utils as utils

def graph():
    loss = []
    test_loss = []
    with open('D:\\paintings\\plots\\output13.txt', 'r') as out:
        for line in list(out):
            if len(line.split()) > 6 and line.split()[4] == 'Test':
                loss.append(line.split()[3])
                test_loss.append(line.split()[6])
    print(len(loss))
    print(len(test_loss))
    graph, (plot1, plot2) = plt.subplots(1, 2)
    plot1.plot(loss, label='loss (training data)')
    plot2.plot(test_loss, label='loss (validation data)')
    plot1.invert_yaxis()
    plot2.invert_yaxis()
    plot1.set_title('loss (training data)')
    plot2.set_title('loss (validation data)')
    plt.title('loss on attention model for Wikiart')
    plt.ylabel('loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    graph.tight_layout()
    plt.show()
    # plt.savefig('D:\\paintings\\plots\\plots1.png')


def graph2():
    loss = []
    test_loss = []
    with open('D:\\paintings\\plots\\output20.txt', 'r') as out:
        for line in list(out):
            if len(line.split()) > 6 and line.split()[4] == 'Test':
                loss.append(float(line.split()[3]))
                test_loss.append(float(line.split()[6][:-1]))
    plt.plot(loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.title('loss on attention model for Wikiart')
    plt.legend()
    plt.show()


def graph_trans():
    loss = []
    test_loss = []
    with open('D:\\paintings\\trans_plots\\realism_trans_cp_lr7_drop0.2_heads16_n6_batch8.txt', 'r') as out:
        for line in list(out):
            line = line.split()
            if line[2] == 'Loss':
                loss.append(float(line[3]))
            elif line[2] == 'TestLoss':
                test_loss.append(float(line[3]))
    plt.plot(loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.title('loss on transformer model for Wikiart frac 0.8')
    plt.legend()
    plt.show()


def graph_history():
    with open('C:\\Users\\anke\\PycharmProjects\\pythonProject\\save4\\histories_.pkl', 'rb') as f:
        history = utils.pickle_load(f)
    # print(history.keys())
    loss = list(map(float, history['loss_history'].values()))
    val_loss = [history['val_result_history'][i]['loss'] for i in history['val_result_history']]
    print(val_loss)
    # print(loss)
    # print(history['loss_history'])
    # x = [i * len(loss) // 4 for i in range(4)]
    plt.plot(loss, label='loss')
    # plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.show()


def preds_scores():
    with open('vis\\test4_test_folder', 'rb') as f:
        preds = utils.pickle_load(f)
    fig = plt.figure()
    for i in range(12, 18):
        item = preds[i]
        ax = fig.add_subplot(2, 3, i + 1 - 12)
        cap = item['caption']
        ax.set_title(cap)
        real_name = item['file_name'].split('\\')[-1]

        img = mimage.imread(item['file_name'].replace('F', 'E', 1))
        imgplot = plt.imshow(img, label=cap)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    preds_scores()




