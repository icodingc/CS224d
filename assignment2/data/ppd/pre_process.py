import itertools
import nltk
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read
print "Read txt file..."

with open("test.txt","rb") as f:
	reader = f.readlines()
	# Split 
	sentences = 