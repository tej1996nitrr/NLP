#%%
import nltk
from  nltk.book import *

# %%
text1

# %%
text1.concordance('monstrous')

# %%
text3.concordance('lived')

# %%
text1.similar('monstrous')

# %%
text2.similar("monstrous")

# %%
text2.common_contexts(["monstrous", "very"])

# %%
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"]) 

# %%
text3.generate()


# %%
len(text1)#no of words

# %%
sorted(set(text3))
print(len(text3))
len( sorted(set(text3)))
# %%
text3.count("smoke")

# %%
def lexical_diversity(text):
    return len(text)/len(set(text))

def percentage(count, total):
    return 100 * count / total 

print(lexical_diversity(text3) )
print(lexical_diversity(text5) )
print(percentage(4, 5))
print(percentage(text4.count('a'), len(text4)) )
# %%
from nltk.probability import FreqDist
fdist1 = FreqDist(text1)
fdist1 
vocabulary1 = fdist1.keys()
print(vocabulary1)
print(fdist1['whale'])

# %%
fdist1.plot(50, cumulative=True)

# %%
