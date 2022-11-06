# TODO: min_split uslov valjda sadrzi vec min_node_dp? proveri

import numpy as np  # must be included
from Node import *  # must be included

# GI is the same as Gini Index

class CaRT():
    """
    Classification and Regression Tree class
    .===========.
    Constructor arguments:
    tensor X - training data
    vector y - training labels, must not be one hot encoded
    int max_depth = 5 - maximum tree depth
    int min_split = 2 - minimum amount of datapoints left after split
    int min_node_dp = 2 - minimum amount of datapoints in node

    Usage:
    >>> clf = CaRT(X, y)
    >>> clf.build_tree()
    >>> clf.predict(testX, clf.root)
    """

    def __init__(self, X, y, max_depth = 5, min_split = 2, min_node_dp = 2):
        """
        Constructs the class with passed values
        Additionaly makes parameters:
        int classes - number of classes in training labels
        Node Root - the root node of the tree
        """

        self.X = X
        self.y = y
        self.classes = np.unique(self.y)

        self.max_depth = max_depth
        self.min_split = min_split
        self.min_node_dp = min_node_dp

        self.Root = None # tree not made yet
    #========

    
    def calc_probs(self, y):
        """
        Calculates the probability to randomly pick a certain class
        using Laplace probability definition: number_of_class_elements / number_of_elements

        Used for calculating Gini Index
        """

        probs = []

        for one_class in self.classes:
            p = y[y == one_class].shape[0] / y.shape[0]
            probs.append(p)

        return np.asarray(probs)
    #========


    def Gini(self, y):
        """
        Calculates Gini Index given probabilities of a certain class
        """
        return 1 - np.sum(self.calc_probs(y)**2)
    #========

    
    def Best_split(self, node_X, node_y):
        """
        Calculates what the best split of the node is based on the Gini Index

        """

        best_feature = None # which feature is the best to make the split on
        best_border = None  # the concrete value of that feature on which the split is made
        best_Gain = -100    # just a small number serving as an error code, Gini interval is [0, 1]

        node_Gini = self.Gini(node_y) # calculating the current Gini Index of the node to compare to split GI
        #-----

        #TODO: ova 2 fora ubrzaj
        # finding the split
        for feature in range(node_X.shape[1]):  # going through every feature
            x_column = node_X[:, feature]       # taking out the collumn of the feature

            for val in x_column:    # for each value in the collumn
                border = val        # im just naming the val border for readability purposes     

                y_right = node_y[x_column >= border]    # picking the labels of the datapoints that satisfy the border
                y_left = node_y[x_column < border]

                if y_right.shape[0] == 0 or y_left.shape[0] == 0: # if none were taken just dont mind it
                    continue

                Gini_left = self.Gini(y_left)   # calculate gini for the split labels
                Gini_right = self.Gini(y_right)

                left_prob = y_left.shape[0] / node_y.shape[0]   # calculate the probabilities to pick left labels out of all labels
                right_prob = y_right.shape[0] / node_y.shape[0] # same for right
                
                split_Gain = node_Gini - (left_prob * Gini_left + right_prob * Gini_right) # calculate information gain

                if split_Gain > best_Gain: # if the information gain just made is better than the best thus far
                    best_feature = feature # make this one the best...
                    best_border = border
                    best_Gain = split_Gain

        if best_Gain == -100: # if no split is made return None for all
            return None, None, None, None, None, None

        # splitting the datapoints and labels by the best border
        x_column = node_X[:, best_feature]

        x_right = node_X[x_column >= best_border]
        x_left = node_X[x_column < best_border]

        y_right = node_y[x_column >= best_border]
        y_left = node_y[x_column < best_border]

        return best_feature, best_border, x_right, x_left, y_right, y_left
    #========

    
    def build_node(self, node_X, node_y, node):
        """
        Recursively builds nodes minding recursion termination
        """

        if node.depth >= self.max_depth:    # checking if the node is above the alowed depth
            node.is_leaf = True             # if not, make it a leaf
            return                          # and terminate recursion for it

        if node_X.shape[0] < self.min_node_dp:  # checking if ther are enough datapoints in the node 
            node.is_leaf = True                 # if not, then its a leaf
            return

        if np.unique(node_y).shape[0] == 1:     # one class left in labels means the node is a leaf
            node.is_leaf = True
            return
        
        feature, border, x_right, x_left, y_right, y_left = self.Best_split(node_X, node_y) # splitting the node

        if feature == None:     # if there is no good split found (error code from Best_split returns all none)
            node.is_leaf = True # then it must be a leaf because there is no better split than the current
            return
            
        if x_right.shape[0] < self.min_split or x_left.shape[0] < self.min_split:   # checking for minimum datapoints split off
            node.is_leaf = True
            return

        node.feature = feature  # defining this nodes parameters returned by Best_split
        node.border = border

        node.right = Node()                         # initializing the left and right nodes and their depths and probabilities
        node.right.depth = node.depth + 1
        node.right.probs = self.calc_probs(y_right) # probabilities are for classification and GI calculation

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probs = self.calc_probs(y_left)

        self.build_node(x_right, y_right, node.right) # recursively building left and right node
        self.build_node(x_left, y_left, node.left)
    #========

    
    def build_tree(self):
        """
        Initializes root and builds the tree recursively from it
        """

        self.Root = Node()
        self.Root.depth = 1
        self.Root.probs = self.calc_probs(self.y)

        self.build_node(self.X, self.y, self.Root)
    #========


    def predict(self, x, node):
        """
        Predicts the class of a singular datapoint by recursively going through the tree
        Usage:
        >>> obj.predict(datapoint, obj.Root)

        returns the probabilities of the datapoint being each of the classes
        """

        if node.is_leaf == True:
            return node.probs

        if x[node.feature] >= node.border:
            probs = self.predict(x, node.right)
        else:
            probs = self.predict(x, node.left)

        return np.asarray(probs)
    #========


    def predict_set(self, X):
        """
        Predicts the classes of a whole dataset using the predict() method
        Usage:
        >>> obj.predict_set(dataset)

        returns an array of probabilities of classes for each datapoint in the set
        """
        pred = []

        for x in X:
            pred.append(self.predict(x, self.Root))

        return np.asarray(pred)
    #========


    def accuracy(self, X, y):
        """
        Prints the accuracy of the model on the given datapoints X and their labels y
        """
        pred = self.predict_set(X)
        ctr = 0

        for i in range(pred.shape[0]):
            y_hat = np.argmax(pred[i])
            if y_hat == y[i]:
                ctr += 1

        print("Model accuracy: {:.2f}%".format((ctr / y.shape[0])*100))
    #========

    
    def print_tree(self, node):
        """
        Recursively prints the data of the tree, starting from the leafs
        """

        if node.is_leaf == True:
            node.print_data()
            return

        self.print_tree(node.right)
        self.print_tree(node.left)

        node.print_data()
    #========