import os, sys
import json
import numpy as np
from absl import app
from absl import flags
import glob
import matplotlib.pyplot as plt
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("root", "", "root folder")
flags.DEFINE_string("subfolder", "" , "subfolder")

class TrainDataLoader:
    def __init__(self, root, subfolder):
        self.root = root
        self.subfolder = subfolder
        self.train_inputs = []
        self.train_outputs = []

    def parse_json(self, fname):
        data = json.load(open(fname))
        train_list = data['train'] + data['test']
        case_inputs = []
        case_outputs = []
        for train_data in train_list:
            train_input = np.array(train_data['input'])
            train_output = np.array(train_data['output'])
            case_inputs.append(train_input)
            case_outputs.append(train_output)
        return case_inputs, case_outputs

    def plot_matrix(self, inputs, outputs):
        n = len(inputs)
        f, axs = plt.subplots(3,n, sharey=True)

        for i in range(n):
            axs[0,i].imshow(inputs[i])
            axs[1,i].imshow(outputs[i])
            axs[2,i].imshow(outputs[i] - inputs[i])
        plt.show()

    def parse_all_jsons(self):
        jsons = glob.glob(os.path.join(self.root, self.subfolder, '*.json'))
        for fjson in jsons:
            inputs, outputs = self.parse_json(fjson)
            self.train_inputs.append(inputs)
            self.train_outputs.append(outputs)

    def random_choose_plot(self):
        matrixs = [ i for i, (x, y) in enumerate(zip(self.train_inputs, self.train_outputs)) if x[0].shape == y[0].shape]
        ind = random.choice(matrixs)
        self.plot_matrix(self.train_inputs[ind], self.train_outputs[ind])


    def print_statistics(self):
        print("-----shape analysis-----")
        print("# {} input output pairs ".format(len(self.train_outputs)))
        # same_shapes = [ 1 if x.shape == y.shape else 0 for (x,y) in zip(self.train_inputs, self.train_outputs)]
        # print("# {} have same shapes for input output ".format(np.sum(same_shapes)))
        #
        # bigger_shapes = [1 if x.shape[0] > y.shape[0] else 0 for (x,y) in zip(self.train_inputs, self.train_outputs)]
        # print("# {}  input shapes > output shape".format(np.sum(bigger_shapes)))
        # smaller_shapes = [1 if x.shape[0] < y.shape[0] else 0 for (x,y) in zip(self.train_inputs, self.train_outputs)]
        # print("# {}  input shapes < output shape".format(np.sum(smaller_shapes)))

def main(argv):
    train_data_loader = TrainDataLoader(FLAGS.root, FLAGS.subfolder)
    train_data_loader.parse_all_jsons()
    train_data_loader.print_statistics()
    train_data_loader.random_choose_plot()

if __name__ == '__main__':
    app.run(main)
