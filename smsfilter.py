import sys
import nltk
import sklearn
import pandas as pd
import numpy as np

# for loading the dataset 
df = pd.read_table('SMSSPamCollection', header=None, encoding='utf-8')
#print(df.info())
#print(df.head())
classes = df[0]
#print(classes.value_counts()) # got 4825 ham messages and 747 spam messages 
from sklearn.preprocessing import LabelEncoder
# giving binary values to the class labels, 0 = ham and 1 = spam
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
#print(Y[:10])
text_messages = df[1]

#replacing email addresses, URLs, phone numbers, other numbers with regular expressions
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation, whitespace between terms, and leading and trailing white space
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')

# changed words to lower case
processed = processed.str.lower()

from nltk.corpus import stopwords
# removing stop words 
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Removing word stems using a Porter stemmer
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

#Creating bag-of-words
from nltk.tokenize import word_tokenize
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)
#print('Number of words: {}'.format(len(all_words)))
#print('Most common words: {}'.format(all_words.most_common(15)))

# used 1500 most common words as features
word_features = list(all_words.keys())[:1500]
#define a find_features function
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
return features

# find features for all the messages
messages = zip(processed, Y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)
# call find_features function for each message
featuresets = [(find_features(text), label) for (text, label) in messages]

# splitting testing and training sets using sklearn
from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
#print(len(training))
#print(len(testing))

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

model.train(training)
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))

# Scikit-learn Classifiers with NLTK
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Defining the models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

# Initializing the classifiers
classifiers = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(),
    SGDClassifier(max_iter = 100),MultinomialNB(),SVC(kernel = 'linear')]
models = zip(names, classifiers)

# Wrap models in NLTK
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))
    # K-Nearest Neighbours: Accuracy: 92.74
    # Decision Tree: Accuracy: 94.83
    # Random-forest: Accuracy: 95.69
    # Logistic Regression: Accuracy: 94.75
    # SGD Classifier: Accuracy: 94.97
    # Naive-Bayes: Accuracy: 95.84
    # SVM Linear: Accuracy: 94.97
    
# Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(),
    SGDClassifier(max_iter = 100),MultinomialNB(),SVC(kernel = 'linear')]

models = zip(names, classifiers)

# Used Hard Voting
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)

# Printed Accuracy
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))
# Got 95.55%

# make class label prediction for testing set
txt_features, labels = zip(*testing)  #unzipping

# printed a classification report
prediction = nltk_ensemble.classify_many(txt_features)
print(classification_report(labels, prediction))

# Plotted a confusion matrix
pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])
