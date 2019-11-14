

    
import numpy as np

np.random.seed(804730892)


class BagLearner(object):

    #initialise learner
    #this is a bag learner which utilises "bagging" of training data or random sampling
    #in the training of a model
    
    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []
        for _ in range(self.bags):
            self.learners.append(learner(**kwargs))


    def author(self):
        return "nwatt3"

    def addEvidence(self, Xtrain, Ytrain):
 #add training data
        #randomly select data using index of the data
        #add the randomly "bagged" data to model
        for learner in self.learners:
            indices = np.arange(Xtrain.shape[0])
            bagging = np.random.choice(indices, size=Xtrain.shape[0], replace=True)
            Xtrain = Xtrain[bagging, :]
            Ytrain = Ytrain[bagging]
            learner.addEvidence(Xtrain, Ytrain)

 
    def query(self, data):
        #for a group of learners, create array of output predictions
        # these are uses as vote
        #mean of the "votes" becomes the ultimate answer
        predictions = []
        for learner in self.learners:
            predictions.append(learner.query(data))
        return np.vstack(predictions).mean(axis=0).reshape(-1)
