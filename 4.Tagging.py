#%%
"""
CC-coordinating conjunction
RB -adverbs
IN-preposition
NN-Noun
JJ-adjective
VBP-present tense verb

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
