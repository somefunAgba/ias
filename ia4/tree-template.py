"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree



###################


def entropy(target_col):
    """
    This function takes target_col, which is the data column containing the class labels, and returns H(Y).

    """
    
    ######################
    #Filling in this part#
    ######################
    return entropy


###################

###################


def InfoGain(data,split_attribute_name,target_name="class"):
    """
    This function calculates the information gain of specified feature. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
        
    ######################
    #Filling in this part#
    ######################

    return Information_Gain

###################

###################


def DecisionTree(data,features,target_attribute_name="class", depth, maxdepth):
    """
    This function takes following paramters:
    1. data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    2. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature

    3. target_attribute_name = the name of the target attribute
    4. depth = the current depth of the node in the tree --- this is needed to remember where you are in the overall tree
    5. maxdepth =  the stopping condition for growing the tree

    """
    #First of all, define the stopping criteria here are some, depending on how you implement your code, there maybe more corner cases to consider
    """
    1. If max depth is met, return a leaf node labeled with majority class, additionally
    2. If all target_values have the same value (pure), return a leaf node labeled with majority class 
    3. If the remaining feature space is empty, return a leaf node labeled with majority class
    """
    ######################
    #Filling in this part#
    ######################


    #If none of the above holds true, grow the tree!
    #First, select the feature which best splits the dataset

    ######################
    #Filling in this part#
    ######################
    
    
    #Once best split is decided, do the following: 
    """
    1. create a node to store the selected feature 
    2. remove the selected feature from further consideration
    3. split the training data into the left and right branches and grow the left and right branch by making appropriate cursive calls
    4. return the completed node
    """
    ######################
    #Filling in this part#
    ######################

###################

###################




def predict(example,tree,default = 1):
    """
    This function handles making prediction for an example, takes two parameters:
    1. The example

    2. The tree, which is a node
    This needs to be done in a recursive manner. First check if the node is a leaf, if so, return the prediction of the node. Otherwise, send the example down the appropriate subbranch with recursive call.
    """


