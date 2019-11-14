import numpy as np 


np.random.seed(666930280)

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        if leaf_size < 1:
            raise Exception("Must have at least one leaf. It's a tree!")
        self.leaf_size = leaf_size


    def author(self):
        return "nwatt3"

    
    
    
    
    def train_tree(self, Xtrain, Ytrain):


        if np.unique(Ytrain).shape[0] == 1:
            
            #if the data has only one yvalue then this is returned as 1 leaf
            
            return np.asarray([[-1, Ytrain[0], np.nan, np.nan]])
        
        
        if self.leaf_size >= Xtrain.shape[0]:
            
            
            #if there is less than leaf size , still return one leaf as lead size has to be 1 at least
            return np.asarray([[-1, np.mean(Ytrain), np.nan, np.nan]])
        
        #this time using a random feature for splitting
        bestFeature = np.random.choice(np.arange(Xtrain.shape[1]))
        split_threshold = np.median(Xtrain[:, bestFeature])
        left_values = Xtrain[:, bestFeature] <= split_threshold
        
        

        # identify best feature to split based on correlation with target Ytrain
       # correlation_array = []
        #for i in range(Xtrain.shape[1]):
            
            
            #variance_matrix = np.var(Xtrain[:, i])
            #correlation_matrix = np.corrcoef(Xtrain[:, i], Ytrain)[0, 1] if variance_matrix > 0 else 0
            #correlation_array.append(correlation_matrix)
        #bestFeature = np.argsort(correlation_matrix)[::-1][0]

        # split threshold for splitting
        #I'm basing this on the median of best feature
        #split_threshold = np.median(Xtrain[:, bestFeature])
        
        
        #left tree defined as less than or equal to the threshold
        #left_values = Xtrain[:, bestFeature] <= split_threshold

        if np.median(Xtrain[left_values, bestFeature]) == split_threshold:
            
            
            # only makes sense to split on a feature if x actually varies
            
            return np.asarray([[-1, np.mean(Ytrain), np.nan, np.nan]])

        
        
        # train right tree, this is just the opposite of left. so used ~
        RHS_tree = self.train_tree(Xtrain[~left_values], Ytrain[~left_values])
        
        
        # train tree left
        LHS_tree = self.train_tree(Xtrain[left_values], Ytrain[left_values])

        #root_node node of the tree
        root_node = np.asarray([[bestFeature, split_threshold, 1, LHS_tree.shape[0]+1]])

        # create the tree with root, LHS and right
        return np.vstack((root_node, LHS_tree, RHS_tree))


    def query_tree(self, tree, Xtrain):
        root_node = tree[0]
        if int(root_node[0]) == -1:
            #return leaf value
            return root_node[1]
        elif Xtrain[int(root_node[0])] <= root_node[1]:
            # left tree
            LHS_tree = tree[int(root_node[2]):,:]
            return self.query_tree(LHS_tree, Xtrain)
        else:
            # go right if value is neither equal to or less than split value
            RHS_tree = tree[int(root_node[3]):,:]
            return self.query_tree(RHS_tree, Xtrain)


    def addEvidence(self, Xtrain, Ytrain):

        #add x train and y train data to model
        
        self.tree = self.train_tree(Xtrain, Ytrain)


    def query(self, Xtest):

        
        Y = []
        for X in Xtest:
            Y.append(self.query_tree(self.tree, X))
        return np.asarray(Y)


