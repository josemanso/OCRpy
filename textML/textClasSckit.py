# text classification usin sckit-learn and nltk

# loading data set from http://qwone.com/~jason/20Newsgroups/

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# Note: Above, we are only loading the training data,
# We will load the test data separately later in the example.

# Check the target names (categories) and some data files.
print(twenty_train.target_names) # print all the categories

# prints first line of the first data file
print("\n".join(twenty_train.data[0].split("\n")[:3]))

# We will be using bag of words model. Briefly, we segment each text file into
# words (for English splitting by space), and count# of times
# each word occurs in each document and finally assign each word
# an integer id. Each unique word in our dictionary will correspond
# to a feature (descriptive feature).

# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(twenty_train.data)
print('n_samples, n_features: ',x_train_counts.shape) # (11314 samples, 130107 features)

# TF  count the number of words in each document
# TF-IDF Term Frequency times inverse document frequency.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print('dimension of the Document-Term matrix: ',x_train_tfidf.shape) # (11314, 130107)

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train_tfidf, twenty_train.target)

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbioing forward.
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Performance of NB Classifier
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle= True)
predicted = text_clf.predict(twenty_test.data)
print('accuracy we get is ~77.38%:  ',np.mean(predicted == twenty_test.target))

# Training Support Vector Machines - SVM and calculating its performance
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf_svm',
                          SGDClassifier(loss='hinge',
                                        penalty='l2',
                                        alpha=1e-3,
                                        max_iter=10,
                                        random_state=42))
                         ])
text_clf_svm = text_clf_svm.fit(twenty_train.data,twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
print('accuracy we get is ~82.38%:  ',np.mean(predicted_svm == twenty_test.target))

# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}

# Next, we create an instance of the grid search by passing the classifier, parameters 
# and n_jobs=-1 which tells to use multiple cores from user machine
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# To see the best mean score and the params, run the following code

print('best_score: ',gs_clf.best_score_)
print('best_params: ',gs_clf.best_params_)

# Output for above should be: The accuracy has now increased to ~90.6% for the NB classifier (not so naive anymore! 😄)
# and the corresponding parameters are {‘clf__alpha’: 0.01, ‘tfidf__use_idf’: True, ‘vect__ngram_range’: (1, 2)}.

# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])

#