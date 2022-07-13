# We can use documents from the nltk.corpus.  As an example, lets load the universal declaration of human rights.
import nltk
#nltk.download()
from nltk.corpus import udhr
print(udhr.raw('English-Latin1'))


# In[11]:


# Lets import some sample and training text - George Bush's 2005 and 2006 state of the union addresses. 

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


# In[12]:


# Now that we have some text, we can train the PunktSentenceTokenizer

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

from spellchecker import SpellChecker


# In[13]:

from nltk.stem import PorterStemmer

ps = PorterStemmer()

# Now lets tokenize the sample_text using our trained tokenizer

tokenized = custom_sent_tokenizer.tokenize(sample_text)

words = tokenized

for w in words:
    print(ps.stem(w))
# In[14]:


# This function will tag each tokenized word with a part of speech

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))

        
# The output is a list of tuples - the word with it's part of speech
process_content()


spell = SpellChecker()
def correct_spellings(sample_text):
    corrected_text = []
    misspelled_words = spell.unknown(sample_text.split())
    for words in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)



# ##### Chunking with NLTK
# 
# Now that each word has been tagged with a part of speech, we can move onto chunking: grouping the words into meaningful clusters.  The main goal of chunking is to group words into "noun phrases", which is a noun with any associated verbs, adjectives, or adverbs. 
# 
# The part of speech tags that were generated in the previous step will be combined with regular expressions, such as the following:

# In[15]:


'''
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line
'''


# In[16]:



def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part-of-speech tag with a regular expression
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # draw the chunks with nltk
            # chunked.draw()     

    except Exception as e:
        print(str(e))

        
process_content()


# The main line in question is:

# In[17]:


'''
chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
'''


# This line, broken down:

# In[18]:


'''
<RB.?>* = "0 or more of any tense of adverb," followed by: 

<VB.?>* = "0 or more of any tense of verb," followed by: 

<NNP>+ = "One or more proper nouns," followed by 

<NN>? = "zero or one singular noun." 

'''


# In[19]:


# We can access the chunks, which are stored as an NLTK tree 

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part-of-speech tag with a regular expression
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
            
            # draw the chunks with nltk
            # chunked.draw()     

    except Exception as e:
        print(str(e))

        
process_content()


# ##### Chinking with NLTK
# 
# Sometimes there are words in the chunks that we don't won't, we can remove them using a process called chinking.

# In[20]:


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # The main difference here is the }{, vs. the {}. This means we're removing 
            # from the chink one or more verbs, prepositions, determiners, or the word 'to'.

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            # chunked.draw()

    except Exception as e:
        print(str(e))

        
process_content()


# ##### Named Entity Recognition with NLTK
# 
# One of the most common forms of chunking in natural language processing is called "Named Entity Recognition." NLTK is able to identify people, places, things, locations, monetary figures, and more.
# 
# There are two major options with NLTK's named entity recognition: either recognize all named entities, or recognize named entities as their respective type, like people, places, locations, etc.
# 
# Here, with the option of binary = True, this means either something is a named entity, or not. There will be no further detail.

# In[21]:


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            # namedEnt.draw()
            
    except Exception as e:
        print(str(e))

        
process_content()


# ### Text Classification
# 
# ##### Text classification using NLTK
# 
# Now that we have covered the basics of preprocessing for Natural Language Processing, we can move on to text classification using simple machine learning classification algorithms.

# In[22]:


import random
import nltk
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

print('Number of Documents: {}'.format(len(documents)))
print('First Review: {}'.format(documents[1]))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print('Most common words: {}'.format(all_words.most_common(15)))
print('The word happy: {}'.format(all_words["happy"]))


# In[23]:


# We'll use the 4000 most common words as features
print(len(all_words))
word_features = list(all_words.keys())[:4000]


# In[25]:


# The find_features function will determine which of the 3000 word features are contained in the review
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Lets use an example from a negative review
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
for key, value in features.items():
    if value == True:
        print (key)


# In[26]:


# Now lets do it for all the documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[27]:


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# define a seed for reproducibility
seed = 1

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[28]:


print(len(training))
print(len(testing))


# In[29]:


# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))