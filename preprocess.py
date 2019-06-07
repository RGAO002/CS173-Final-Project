import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

### Input: f_in: str, path for file.
### Outputs: 
### 	bigram: dict, contains bigrams and frequencies in form as '(term_1, term_2): value'.
### 	trigram: dict, contains trigrams friquencies in form as '(term_1, term_2, term_3): value'.
### 	n_bigram: int, number of bigrams.
### 	n_trigram: int, number of trigrams.

def processing(f_in):
	with open(f_in, 'r') as r:
		data = csv.reader(r)
		data = list(data)
		data.pop(0)
	data = [data[x][1].strip() for x in range(len(data))]
	stemmer = PorterStemmer()
	tokenizer = RegexpTokenizer(r'\w+')
	bigram = {}
	trigram = {}
	for line in data:
		words = tokenizer.tokenize(line)
		words = [stemmer.stem(words[x].lower()) for x in range(len(words))]
		for i in range(len(words)):
			if i == len(words) - 1:
				t_next = ''
			else: t_next = words[i + 1]
			if i >= len(words) - 2:
				t_third = ''
			else: t_third = words[i + 2]
			if not bigram.__contains__((words[i], t_next)):
				bigram[words[i], t_next] = 0
			bigram[words[i], t_next] += 1
			if not trigram.__contains__((words[i], t_next, t_third)):
				trigram[words[i], t_next, t_third] = 0
			trigram[words[i], t_next, t_third] += 1
	n_bigram = len(bigram)
	n_trigram = len(trigram)
	for key in bigram:
		bigram[key] /= n_bigram
	for key in trigram:
		trigram[key] /= n_trigram
	return bigram, trigram, n_bigram, n_trigram

### for testing
'''
f_in = 'abcnews-date-text.csv'
(bigram, trigram, n_bigram, n_trigram) = processing(f_in)
print(n_bigram, n_trigram)
print(list(bigram.items())[:10], list(trigram.items())[:10])
'''