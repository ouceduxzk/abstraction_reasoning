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

    First, we need to build basic understandings of input or output array
        1. # of blocks by connected componenet
        2. geometry of the block,  rectangle , square or non-uniform ?
        3. colors inside the block, uniform or multi-color.
        4. areas of each block

    Secondly, build comparison between input and output
        1. which part of each block is changed,
        2. what are the changes?  adds-on, in-place, shifted, overwrite
        3. which color got changed which color not changed w.r.t each block
        4. what are the position, direction relationship between the starting point of adds-on block w.r.t closenst
            original block.

    Thirdly, Develop corresponding policy

        1. determine which types of transformation happen,  regional or global ?
        2. if regional, where are the operations operates on and what kind of operation ?
            2.1 in-place , learn color mapping policy
            2.2 add-on and complex operation with one connected path
                2.2.1. determine the starting and ending position
                2.2.2  learn the policy that which direction to go and which color to map with
                    states are { all colors }
                    actions are [left, up, down, right, left-up, left-down, right-up, right-down]
            2.3 adds-on and complex operations with multiple colors or multiple paths
                2.3.1 determine starting and ending point
                2.3.2 divide each block region into subblock based on color or connectnedness,  apply 2.2.2

            2.4 shift with only 1 color for the whole block
                2.4.1 determine the translation vector
            2.5 adds-on not each pixel but each block, learn policy to shift a block

        3. if global,  what kind of operations ?
            3.1 op that perserve orientation of block
                3.1.1 color changed (# of color doesn't chnage) while shape and areas perserved,
                      then sort based on position, area and find the rule.
                3.1.2 color changed (# of color from 1 increased to 2) use any changed color position as the
                      starting point and need to visit all other pos with same colors such that
                      it learns the policy to go left, right , up, down for those simliar input
                      scenarios. for example,
            3.2 op that changes orientation of block
                3.2.1 rotate , reflection

 '''

import numpy as np

def sum_diff_zero(arr1 : np.array, arr2 : np.array) -> bool:
    return np.sum(arr1 - arr2) == 0

def background_diff_zero(arr1 : np.array, arr2:  np.array) -> bool:
    ind1 = np.where(arr1 == 0)
    ind2 = np.where(arr2 == 0)
    return ind1 == ind2
