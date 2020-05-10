import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from rnn_class.brown import get_sentences_with_word2idx_limit_vocab
    
class SoftmaxANN(BaseEstimator, ClassifierMixin):

    """
    one-layer ANN: W ia a feature_dim * output_dim matrix.
    K is the number of possible targets (i.e. output_dim)
    """
    def __init__(self, K, start_idx=None, end_idx=None):
        self.K = K
        self.start_idx = start_idx
        self.end_idx = end_idx

    def softmax(self, A):
        """ Column-wise softmax calculation."""
        expA = np.exp(A - A.max())
        return expA / expA.sum(axis=1, keepdims=True)
        
    def forward(self, X):
        """ W dim       : feature_dim * output_dim.
            returned dim: sample_size * output_dim
        """
        A = X.dot(self.W)
        Y_prob = self.softmax(A)
        return Y_prob
    
    def indicator(self, y):
        """ turn y into an indicator matrix Y with dim shape
            dim: (N - sample_size, K - output_dim)
        """
        N = len(y)
        Y = np.zeros([N, self.K])
        Y[np.arange(N), y] = 1
        return Y
    
    def cost(self, Y_prob, Y):
        """ Calculate the normalized cross-entropy 
            of target and predicion
        """
        return -np.sum(Y * np.log(Y_prob)) / len(Y_prob)
    
    def get_costs(self):
        return self.costs
    
    def _fit_sentence(self, X, y, epochs, lr, freq):
        """ X is an array of sentences
        """
        random.shuffle(X)
        
        for epoch in range(epochs):
            i = 0
            
            for sentence in X:
                inputs = [self.start_idx] + sentence
                targets = sentence + [self.end_idx]
                
                # Convert into indicator matrix
                inputs = self.indicator(inputs)
                targets = self.indicator(targets)
            
                # Get prediction
                Y_prob = self.forward(inputs)
                
                # Conduct gradient descent
                self.W = self.W - lr * inputs.T.dot(Y_prob - targets)
                
                c = self.cost(Y_prob, targets)
                self.costs[epoch] = c
            
                if i % freq == 0:
                    print("epoch:", epoch, 
                          "sentence: %s/%s" % (i, len(X)), 
                          "cost:", c)
                i += 1
                    
        return self
    
    def _fit_normal(self, X, y, epochs, lr, freq):
        
        Y = self.indicator(y, X.shape(0))
        
        for epoch in range(epochs):
            # Get prediction
            Y_prob = self.forward(X)
        
            # Conduct gradient descent
            self.W = self.W - lr * X.T.dot(Y_prob - Y)
            
            c = self.cost(Y_prob, Y)
            self.costs[epoch] = c
            
            if epoch % freq == 0:
                print("epoch:", epoch, "cost:", c)
            
            # gradient descent
            self.W = self.W - lr * X.T.dot(Y_prob - Y)

        return self
        
    def fit(self, X, y=None, D=None, epochs=500, lr=0.01, freq=400, is_sent=False):
        """  """
        # Initialization
        D = X.shape[1] if D is None else D
        self.W = np.random.randn(D, self.K) / np.sqrt(D)    
        self.costs = np.empty(epochs)
        print("W shape", self.W.shape)
        
        if is_sent:
            return self._fit_sentence(X, y, epochs, lr, freq)
        else:
            return self._fit_normal(X, y, epochs, lr, freq)
        

    def predict(self, X):
        """  """
        if self.W is None:
            print("Please train the estimator first!")
            exit
            
        y_prob = self.forward(X)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

def main():

    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    V = len(word2idx)
    print("Vocab size:", V)

    start_idx = word2idx['START']
    end_idx = word2idx['END']
    
    estimator = SoftmaxANN(K=V, start_idx=start_idx, end_idx=end_idx)
    estimator.fit(sentences, D=V, epochs=1, lr=0.1, freq=50, is_sent=True)
    
if __name__ == '__main__':
    main()
    
    
