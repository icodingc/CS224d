#http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt,axis=1)

def make_onehot(i, n):
    y = zeros(n)
    y[i] = 1
    return y

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, vdim, hdim,bptt = 1):
		self.hdim = hdim
		self.vdim = vdim
		self.bptt = bptt

		# Randomly initialize the parameters
		self.U = np.sprt(1./vdim) * np.random.randn((vdim,hdim))
		self.H = np.sprt(1./hdim) * np.random.randn((hdim,hdim))
		self.L = np.sqrt(1./hdim) * np.random.randn((vdim,hdim))

	def forward(self,xs,ys=None):
		ns = len(xs)
        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
		hs = np.zeros((ns+1,self.hdim))
		# predicted
        ps = np.zeros((ns, self.vdim))
        J = 0
        for t in xrange(ns):
        	hs[t] = np.tanh(self.H.dot(hs[t-1]) + self.L[xs[t]])
        	ps[t] = softmax(self.U.dot(hs[t]))
        	if ys is not None:
        		J += -log(ps[t][ys[t]])
        return hs,ps,J

    def predict(self,xs):
    	hs,ps = self.forward(xs)
    	return np.argmax(ps,axis = 1)

    def compute_loss(self,xs,ys):
    	return forward(xs,ys)[-1]

    def total_loss(self,x,y):
    	J = 0
    	for i in xrange(len(y)):
    		J += compute_loss(x[i],y[i])
    	return J/len(y)

    def bptt(self,xs,ys):
    	ns = len(y)
    	hs,ps = self.forward(xs,ys)[:2]
        # Backward propagation through time
        # should consider bptt
        dldu = np.zeros(self.U.shape)
        dldh = np.zeros(self.H.shape)
        dldl = np.zeros(self.L.shape)

        for t in reversed(range(0,ns)):
            dscores= ps[t] - make_onehot(ys[t],self.vdim)
            dldu += outer(dscores,hs[t])

            delta_t = self.params.U.T.dot(dscores) * hs[t] * (1 - hs[t])
            dldh += outer(delta_t,hs[t-1])
            dldl[xs[t]] += delta_t
            
            # should iter to hs[-1]
            for tt in xrange(t-1,max(0,t-self.bptt)-1,-1):
                delta_t = self.params.H.T.dot(delta_t) * hs[tt] * (1 - hs[tt])
                dldh += outer(delta_t,hs[tt-1])
                dldl[xs[tt]] += delta_t
        return dldl,dldu,dldh

