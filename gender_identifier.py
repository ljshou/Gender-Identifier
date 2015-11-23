#!/usr/bin/python

def gender_features(word):
	return { 'last_letter' : word[-1] }
#end


# may lead to overfitting
# too many features, on a small training set
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
 
import random
import nltk 


names = ([(name.strip(),'male') for name in open('male.txt')] + [(name.strip(),'female') for name in open('female.txt')])

random.shuffle(names)
featuresets = [(gender_features(n),g) for (n,g) in names]

from nltk.classify import apply_features 

train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set)

classifier.show_most_informative_features(5)

# print classifier.classify(gender_features('ravi'))
# print classifier.classify(gender_features('neo'))
# print classifier.classify(gender_features('jack'))
# print classifier.classify(gender_features('dexter'))
