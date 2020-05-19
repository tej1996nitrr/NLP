#%%
#name classifier
from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)

#%%
import nltk
def gender_features(word):
    return {'last_letter': word[-1]}
featuresets = [(gender_features(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))
# %%
"""When working with large corpora, constructing a single list that contains the features
of every instance can use up a large amount of memory. In these cases, use the function
nltk.classify.apply_features, which returns an object that acts like a list but does not
store all the feature sets in memory:"""
from nltk.classify import apply_features
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

# %%
nltk.classify.accuracy(classifier, test_set)

# %%
