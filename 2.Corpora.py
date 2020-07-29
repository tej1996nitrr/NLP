#%%
#electronic books
import nltk
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
# %%
len(emma)


# %%
emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")

# %%
#Gutenberg Corpus
"""average word length, average sentence
length, and the number of times each vocabulary item appears in the text on
average ( lexical diversity score)."""
for fileid  in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print (int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab),fileid)

# %%
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
print(macbeth_sentences)

# %%
for fileid in webtext.fileids():
    print( fileid, webtext.raw(fileid)[:65], '...')

# %%
#Web and Chat Text
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[0]
# %%
#Brown Corpus
brown.categories()
news = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m,":",fdist[m])
# %%
cfd = nltk.ConditionalFreqDist((genre,word) for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

# %%
#Reuters Corpus
reuters.fileids()
reuters.categories()
reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids('barley')
reuters.words('training/9865')[:14] #words or sentences we want in terms of files or categories
# %%
#Inaugural Address Corpus
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]
# %%
cfd = nltk.ConditionalFreqDist((target, fileid[:4]) for fileid in inaugural.fileids() for w in inaugural.words(fileid) for target in ['america', 'citizen'] if w.lower().startswith(target))
cfd.plot()

# %%
genre_word = [(genre, word)
 for genre in ['news', 'romance']
 for word in brown.words(categories=genre)]
print(len(genre_word))
print(genre_word[:4])
print(genre_word[-4:])
# %%
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd)
print(cfd.conditions())
print(cfd['news'])
print(cfd['romance'])
# print(list(cfd['romance']))

# %%
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cfd.tabulate(samples=days)
# cfd.plot()
# %%
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
print(nltk.bigrams(sent).__next__())

# %%
stopwords.words('english')
# %%

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
content_fraction(nltk.corpus.reuters.words())

# %%
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters]

# %%
wn.synsets('motorcar') #synset or “synonym set,”
print(wn.synset('car.n.01').lemma_names)
wn.synset('car.n.01').definition
wn.synsets('car')
# %%
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[26]
# sorted([lemma.name for synset in types_of_motorcar for lemma in synset.lemmas])
# %%
wn.synset('whale.n.02').min_depth()

# %%
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms()
# %%
