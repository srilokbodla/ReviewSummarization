import json
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

with open("reviews.json") as json_data:
    data_file= json.load(json_data)

data = []

for i in range(len(data_file)):
   data.append(data_file[i]['reviewText'])

#print data

sentence = []
for i in data:
    sentence.append((nltk.sent_tokenize(i)))

#print sentence

stop = set(stopwords.words('english') + list(string.punctuation))

word_list = []
for i in data:
    word_list.append((nltk.word_tokenize(i)))

useful_words = []

for i in word_list:
    word = []
    for item in i:
        if item not in stop:
            word.append(item)
    useful_words.append(string.join(word, " "))

# print useful_words[0][0]

vector = TfidfVectorizer(max_features=30, ngram_range=(2, 3))
x = vector.fit_transform(useful_words)

ngrams = vector.get_feature_names()

keys = []

keywords = []
for i in range(len(ngrams)):
    keys.append(nltk.pos_tag(nltk.word_tokenize(ngrams[i])))
    if len(set([keys[i][0][1] , keys[i][1][1]]).intersection(['JJ' , 'JJR' , 'JJS' , 'RB' , 'RB' , 'RBR' , 'RBS'])):
        keywords.append(ngrams[i])

#print keys


print keywords
