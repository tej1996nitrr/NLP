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
#developement set
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, devtest_set))

# %%
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

# %%
for (tag, guess, name) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    print ('correct=%-8s guess=%-8s name=%-30s' %(tag, guess, name))

# %%
def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}

# %%
train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, devtest_set))

# %%
