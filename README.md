# 315-project-3
A RNN to handle reddit comments, built for a deep-learning class.

Authors: Griffin Noe and UT Thapa
Csci 315
Project 3 pt 1
11/6/19

To start, we instantiate t (a list of sentences, soon to be tokenized), all_t (a list of every word in
the corpus), and vocab (a list of the 8000 most common words). We then use nltk.corpus to get a list 
of english stopwords. 

We loop through the data (read from the csv). At each new row (sentence) of the csv, we create an array 
that starts with "START". We then tokenize the row (sentence) of the csv and check if that token is a 
stop word. If it is not a stop word, we add it to the array starting with "START". We also append that token
to our all_t list. We then append "END" to our array starting with "START" and then append t with that array.

We then use nltk freqdist and the most_common function on all_t to append the 8000 most popular tokens into 
our vocab list. We also append "START", "END", and "UNK" to vocab. 

We then loop through our t list with a second loop going through the tokens within every sentence within the 
list t. If that token is not in the vocab, we replace it with "UNK".

Finally, we run word2vec on our list t and output the first ten sentences from t as well as their equivalent
vector embedding. 