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
