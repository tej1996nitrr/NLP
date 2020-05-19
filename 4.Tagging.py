#%%
"""
CC-coordinating conjunction
RB -adverbs
IN-preposition
NN-Noun
JJ-adjective
VBP-present tense verb
ADJ     adjective              new, good, high, special, big, local
ADV     adverb              really, already, still, early, now
CNJ     conjunction              and, or, but, if, while, although
DET     determiner              the, a, some, most, every, no
EX     existential              there, there’s
FW     foreign word              dolce, ersatz, esprit, quo, maitre
MOD     modal verb              will, can, would, may, must, should
N     noun              year, home, costs, time, education
NP     proper noun              Alison, Africa, April, Washington
NUM     number              twenty-four, fourth, 1991, 14:24
PRO     pronoun              he, their, her, its, my, I, us
P     preposition              on, of, at, with, by, into, under
TO     the              word to to
UH     interjection              ah, bang, ha, whee, hmpf, oops
V     verb              is, has, get, do, make, see, run
VD     past tense              said, took, told, made, asked
VG     present participle              making, going, playing, working
VN     past participle              given, taken, begun, sung
WH     wh determiner              who, which, when, what, where, how

"""
import nltk
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

# %%
nltk.help.upenn_tagset('RB')
nltk.help.upenn_tagset('JJ')
# %%
text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

# %%
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')

# %%
text.similar('bought')

# %%
text.similar('over')

# %%
text.similar('the')

# %%
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
# %%
sent = """
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./"""
[nltk.tag.str2tuple(t) for t in sent.split()]

# %%
nltk.corpus.brown.tagged_words()
nltk.corpus.nps_chat.tagged_words()

# %%
nltk.corpus.indian.tagged_words()

# %%
# most common tags in the news category of the Brown Corpus
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
alice2 = [mapping[v] for v in alice]
alice2[:100]
# %%
len(set(alice2))

# %%
from nltk.corpus import brown
counts = nltk.defaultdict(int)
pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news')
for (word, tag) in brown.tagged_words(categories='news'):
    counts[tag] += 1

# %%
counts['NP']

# %%

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
#default tagger
"""Default taggers assign their tag to every single word, even words that have never been
encountered before. once we have processed several thousand words of
English text, most new words will be nouns.
this method performs rather poorly. On a typical corpus, it will tag
only about an eighth of the tokens correctly"""
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()

# %%
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)

# %%
default_tagger.evaluate(brown_tagged_sents)

# %%
patterns = [
 (r'.*ing$', 'VBG'), # gerunds
 (r'.*ed$', 'VBD'), # simple past
 (r'.*es$', 'VBZ'), # 3rd singular present
 (r'.*ould$', 'MD'), # modals
 (r'.*\'s$', 'NN$'), # possessive nouns
 (r'.*s$', 'NNS'), # plural nouns
 (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
 (r'.*', 'NN') # nouns (default)
 ]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

# %%
# The Lookup Tagger
import nltk
from nltk.corpus import brown
fd  = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
cfd
# print(cfd.conditions())
# print(cfd['news'])
most_freq_words = fd.keys()
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)
# %%
#N-Gram tagging
#training a unigram tagger, used it to tag a sentence, and then evaluate
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2004])

brown_sents[0]
brown_tagged_sents[0]
unigram_tagger.evaluate(brown_tagged_sents)
# %%
"""A tagger that simply memorized its
training data and made no attempt to construct a general model would get a perfect
score, but would be useless for tagging new text. Instead, we should split the data,
training on 90% and testing on the remaining 10%"""
size = int(len(brown_tagged_sents) * 0.9)
print(size)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

# %%
"""An n-gram tagger is a generalization of a unigram tagger whose context is the current
word together with the part-of-speech tags of the n-1 preceding tokens"""
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])

# %%
bigram_tagger.evaluate(test_sents) #Its overall accuracy score is very low:
"""As n gets larger, the specificity of the contexts increases, as does the chance that the
data we wish to tag contains contexts that were not present in the training data. This
is known as the sparse data problem,"""

# %%
