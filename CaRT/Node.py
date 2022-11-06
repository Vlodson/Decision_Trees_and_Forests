class Node():
    """ Binary tree node for CaRT Decision Tree 
        Has left and right branch links
        Feature, border and probs as data
        Depth and is_leaf as node parameters"""

    def __init__(self):
        """ Initialises all class fields as None except is_leaf which is False """

        #links
        self.right = None
        self.left = None

        # data
        self.feature = None
        self.border = None
        self.probs = None

        # params
        self.depth = None
        self.is_leaf = False # turns True when the node is an end node

    def print_data(self):
        print("feature:", self.feature)
        print("border:", self.border)
        print("probs:", self.probs)
        print("depth:", self.depth)
        print("is_leaf:", self.is_leaf)
        print()