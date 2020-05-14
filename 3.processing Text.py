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
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government. Supreme executive power derives from
 a mandate from the masses, not from some farcical aquatic ceremony."""
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
tokens = nltk.word_tokenize(raw)
print([porter.stem(t) for t in tokens])
print([lancaster.stem(t) for t in tokens])

# %%
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

# %%
raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
 though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
 well without--Maybe it's always pepper that makes people hot-tempered,'..."""
re.split(r' ', raw)

# %%
# The regular expression «[ \t\n]+» matches one or more spaces
re.split(r'[ \t\n]+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*', raw)
# %%
print (re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))

# %%
# \D Any non-digit character (equivalent to [^0-9])
# \s Any whitespace character (equivalent to [ \t\n\r\f\v]
# \S Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
# \w Any alphanumeric character (equivalent to [a-zA-Z0-9_])
# \W Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])
# \t The tab character
# \n The newline character

# %%
