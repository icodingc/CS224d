from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        # any other initialization you need
        self.sparams.L = wv.copy() 
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        #### END YOUR CODE ####



    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        x =  hstack(self.sparams.L[window])
        
        dd = len(x)/3
        a =  self.params.W.dot(x) + self.params.b1
        h =  tanh(a)
        scores = self.params.U.dot(h) + self.params.b2
        p = softmax(scores) 
        ##
        # Backpropagation
        y = make_onehot(label,len(p))
        delta = p - y
        self.grads.U += outer(delta,h) + self.lreg * self.params.U  # 5 *100
        self.grads.b2 += delta
        dh = self.params.U.T.dot(delta)          #100
        da = dh * (1-tanh(a)**2)
        self.grads.W += outer(da,x) + self.lreg * self.params.W      #100*150
        self.grads.b1 = da

        # good
        dx = self.params.W.T.dot(da)
        dx__ = reshape(dx,(3,-1))
        self.sgrads.L[window[0]] = dx__[0]
        self.sgrads.L[window[1]] = dx__[1]
        self.sgrads.L[window[2]] = dx__[2]

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
        n = len(windows)
        P = zeros((len(windows),self.params.b2.shape[0]))
        #### YOUR CODE HERE ####
        for idx in xrange(n):
            window = windows[idx]
            x =  hstack(self.sparams.L[window])
            h = tanh(self.params.W.dot(x) + self.params.b1)
            scores = self.params.U.dot(h) + self.params.b2
            P[idx,:]= softmax(scores)
        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        p = self.predict_proba(windows)
        c = argmax(p,axis=1)

        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """
        # do not using len(windows)  = window size  
        # because only-one case!!!!
        #### YOUR CODE HERE ####
        pp = self.predict_proba(windows)
        labels2 = reshape(labels,pp.shape[0])
        J = 0.0
        for idx in xrange(pp.shape[0]):
            J  += -1.0 * log(pp[idx,labels2[idx]])
        Jreg = self.lreg/2.0 * (sum(self.params.W**2.0)+sum(self.params.U**2.0))
        #### END YOUR CODE ####
        return J+Jreg