import json
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

with open("reviews.json") as json_data:
    data_file = json.load(json_data)

data = []

for i in range(len(data_file)):
    data.append(data_file[i]['reviewText'])

# print data

# making list of not useful word
stop = set(stopwords.words('english') + list(string.punctuation))

# tokenizing review text into word
word_list = []
for i in data:
    word_list.append((nltk.word_tokenize(i)))

useful_words = []

# removing stopwords

for i in word_list:
    word = []
    for item in i:
        if item not in stop:
            word.append(item)
    useful_words.append(string.join(word, " "))

# print useful_words[0][0]
# tfidfvector is used to assign tfidf value to tokens
# token size can be regulated using ngram_range by default it is 1

vector = TfidfVectorizer(max_features=30, ngram_range=(2, 3))
x = vector.fit_transform(useful_words)

ngrams = vector.get_feature_names()

keys = []

keywords = []
# extracting key words by pos tagging
for i in range(len(ngrams)):
    keys.append(nltk.pos_tag(nltk.word_tokenize(ngrams[i])))
    if len(set([keys[i][0][1], keys[i][1][1]]).intersection(['JJ', 'JJR', 'JJS', 'RB', 'RB', 'RBR', 'RBS'])):
        keywords.append(ngrams[i])

# print keys

# filtering extracted keywords
#print keywords
a = []
duplicateKeys=[]
for i in range(len(keywords)):
    a.append(nltk.pos_tag(nltk.word_tokenize(keywords[i])))
    if len(set([a[i][0][1], a[i][1][1]]).intersection(['PRP', 'IN', 'DT', 'FW', 'LS', 'MD', 'POS', 'SYM', 'TO', 'CD', 'CC', 'UH'])):
        duplicateKeys.append(keywords[i])

keyList=[x for x in keywords if x not in duplicateKeys]
key1 = []
key2 = []
key3 = []

# making list of reviews containing keywords (for each keyword separate list)
dictionary={}
for line in range(len(useful_words)):
    if useful_words[line].find(keyList[0]) != -1:
        key1.append(data_file[line]['reviewText'])
        #dictionary.update({data_file[line]['reviewerID'] , data_file[line]['reviewText']})
        dictionary[data_file[line]['reviewerID']]=data_file[line]['reviewText']
    if useful_words[line].find(keyList[1]) != -1:
        key2.append(data_file[line]['reviewText'])
        dictionary[data_file[line]['reviewerID']] = data_file[line]['reviewText']
    if useful_words[line].find(keyList[7]) != -1:
        key3.append(data_file[line]['reviewText'])
        dictionary[data_file[line]['reviewerID']] = data_file[line]['reviewText']

print len(key1)
print len(key2)
print len(key3)


