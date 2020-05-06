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



    def get_cases(self):
        '''
        1. rotate reflection ----  property : sum = 0
        2. apply rules  based on anchor ----- property : input[nonzero] = output[nonzero at that]
        3. apply rules based on shape --- property :
        4. apply rules based on mask ---  property : diff is only at mask,rule is smoooth in diagonal,
        but how to fix the mask is dependent on the input case
        5. apply rules base on block ---- property : each block is mapped to a new block,location is same, the rule
        can be calculate only on the block in a way that some fixed rule is found and some shape-dependent rule
        is found
        6.apply rules base on block ---- property : each block is mapped to a new block,location is same, rule is
        found based on the geomtry of the block, some block are changed, some are not, sometimes there are only one
        block, sometimes there are multiple, need to understand the geometry of each block

        7. apply rules base on one block -- property : there are background blocks in input not in output, and
        rotate blocks in input subblocks applyed op1 , op2 ... into other background blocks .
        8. different colors of two blocks are digagonal blocks of the output blocks, rule is based on the input block
        are contained in the output blocks and output blocks have one big connected components for each color.
        9.apply rules base on one block with multiple colors : output of other background blocks, not cc , but maybe
        1 step away blocks, chooose those blocks with one color that are equivalent to the original back by rotation.
        then apply the rotation on the two color bigger blocks.
        10 apply rules on the big connected component where the orignial color is changed completely and the closed
        region inside the original block will be changed to the original color. need to do some A* search
        11. apply rules basd on one block with multiple color and then map itself directly to the other blocks and
        delete itself in the original block.

        in summary : we need to build primary operations that are compooble with each other.

        1. find block by connected componenet
        2. get the geometry of the block,  rectange , square or non-uniform ?
        3. analyze colors inside the block, uniform or multi-color.
        4. analyze which part of block is changed in output vs input w.r.t location or colors
        5. as a robot start from the changing region, able to learn the policy that which direction to go and
        which color to map
        6.
         '''
        return

    def parse_all_jsons(self):
        jsons = glob.glob(os.path.join(self.root, self.subfolder, '*.json'))
        for fjson in jsons:
            inputs, outputs = self.parse_json(fjson)
            self.train_inputs.append(inputs)
            self.train_outputs.append(outputs)

    def random_choose_plot(self):
        matrixs = [ i for i, (x, y) in enumerate(zip(self.train_inputs, self.train_outputs)) if x[0].shape == y[0].shape]
        ind = random.choice(matrixs)
        #import pdb; pdb.set_trace()
        print(self.train_inputs[ind][0])
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
