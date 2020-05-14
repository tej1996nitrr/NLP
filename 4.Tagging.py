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
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.keys()


# %%
