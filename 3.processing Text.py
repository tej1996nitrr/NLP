#%%
import re
import nltk
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

# %%
[w for w in wordlist if re.search('ed$', w)]

# %%
# The . wildcard symbol matches any single character
[w for w in wordlist if re.search('^..j..t..$', w)]

# %%
# the ? symbol specifies that the previous character is optional
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]

# %%
# + simply means “one or more instances of the preceding item,”
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]

# %%
# *, which means “zero or more instances of the preceding item.”
[w for w in chat_words if re.search('^m*i*n*e*$', w)]


# %%
wsj = sorted(set(nltk.corpus.treebank.words()))
[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]

# %%
[w for w in wsj if re.search('^[A-Z]+\$$', w)]

# %%
[w for w in wsj if re.search('^[0-9]{4}$', w)]

# %%
[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]

# %%
# ^abc Matches some pattern abc at the start of a string
# . Wildcard, matches any character
# abc$ Matches some pattern abc at the end of a string
# [abc] Matches one of a set of characters
# [A-Z0-9] Matches one of a range of characters
# ed|ing|s Matches one of the specified strings (disjunction
# * Zero or more of previous item, e.g., a*, [a-z]* (also known as Kleene Closure)
# + One or more of previous item, e.g., a+, [a-z]+
# ? Zero or one of the previous item (i.e., optional), e.g., a?, [a-z]?
# {,n} No more than n repeats
# {m,n} At least m and no more than n repeats
# a(b|c)+ Parentheses that indicate the scope of the operators
# {n,} At least n repeats
# {n} Exactly n repeats where n is a non-negative integer
[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]

# %%
[w for w in wsj if re.search('(ed|ing)$', w)]

# %%
