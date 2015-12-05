from cs224d.word2vec import *
from cs224d.SentiAnalysis import *

# Load some data and initialize word vectors

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# # # We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# # Context size
# C = 5

# # Train word vectors (this could take a while!)

# # Reset the random seed to make sure that everyone gets the same results
# random.seed(31415)
# np.random.seed(9265)
# wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors, 
#                               np.zeros((nWords, dimVectors))), axis=0)
# wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), 
#                    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=100)
# # sanity check: cost at convergence should be around or below 10

# # sum the input and output word vectors
# wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

# print "\n=== For autograder ==="
# checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
# checkIdx = [tokens[word] for word in checkWords]
# checkVecs = wordVectors[checkIdx, :]
# print checkVecs

# Visualize the word vectors you trained

_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
# visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"]
# visualizeIdx = [tokens[word] for word in visualizeWords]
# visualizeVecs = wordVectors[visualizeIdx, :]
# temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
# covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
# U,S,V = np.linalg.svd(covariance)
# coord = temp.dot(U[:,0:2]) 

# for i in xrange(len(visualizeWords)):
#     plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
    
# plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
# plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
# plt.show()


# Gradient check always comes first
# random.seed(314159)
# np.random.seed(265)
# dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
# dummy_features = np.zeros((10, dimVectors))
# dummy_labels = np.zeros((10,), dtype=np.int32)    
# for i in xrange(10):
#     words, dummy_labels[i] = dataset.getRandomTrainSentence()
#     dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
# print "==== Gradient check for softmax regression ===="
# gradcheck_naive(lambda weights: softmaxRegression(dummy_features, dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

# print "\n=== For autograder ==="
# print softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0)


# Try different regularizations and pick the best!

### YOUR CODE HERE

regularization = 0.00003 # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best

### END YOUR CODE

random.seed(3141)
np.random.seed(59265)
weights = np.random.randn(dimVectors, 5)

trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)

for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
# We will do batch optimization
weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, weights, regularization), weights, 3.0, 10000, PRINT_EVERY=1000)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)

for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(devFeatures, devLabels, weights)
print "Dev precision (%%): %f" % precision(devLabels, pred)




# Test your findings on the test set

testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)

for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(testFeatures, testLabels, weights)
print "=== For autograder ===\nTest precision (%%): %f" % precision(testLabels, pred)