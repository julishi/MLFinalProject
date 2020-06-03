import glob
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


######
#Get the training data
######
pos_samples = glob.glob("/WAVE/projects/COEN-281-Wi20/juliana/ML/data/imdb/train/pos/*.txt")
neg_samples = glob.glob("/WAVE/projects/COEN-281-Wi20/juliana/ML/data/imdb/train/neg/*.txt")
#print(pos_samples)
training_data = pos_samples + neg_samples
print(training_data)
print(len(training_data))

ones = [1] * len(pos_samples)
negs = [-1] * len(neg_samples)
y_train = ones + negs

######
#Get the testing data
######
pos_samples = glob.glob("/WAVE/projects/COEN-281-Wi20/juliana/ML/data/imdb/test/pos/*.txt")
neg_samples = glob.glob("/WAVE/projects/COEN-281-Wi20/juliana/ML/data/imdb/test/neg/*.txt")
testing_data = pos_samples + neg_samples
print(len(testing_data))

ones = [1] * len(pos_samples)
negs = [-1] * len(neg_samples)
y_test = ones + negs


######
#Vectorize the data
#####
vectorizer = TfidfVectorizer()
vectorizer.fit(training_data)
print("Vectorizing")
X_train = vectorizer.transform(training_data)
print("X train data")
print(X_train)
print(X_train.shape)
X_test = vectorizer.transform(testing_data)
print(X_test.shape)

#Apeksha's Method##
#cv = CountVectorizer()
#word_count_vector = cv.fit_transform(training_data)
#tfidf_transf = TfidfTransformer(smooth_idf=True, use_idf=True)
#tfidf_transf.fit(word_count_vector)
#X_train = tfidf_transf.transform(word_count_vector)

#test_word_count = cv.transform(testing_data)
#X_test = tfidf_transf.transform(test_word_count)
#print("Train/Test shape:")
#print(X_train.shape)
#print(X_test.shape)

####
#Training the models
####
print("Training")
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

####
#Test
####
print("Prediction")
predictions = clf.predict(X_test)

#a = clf.score(X_test, y_test)
#print("Acc:", a)
y_test = np.array(y_test)
print(classification_report(y_test, predictions))

print("Pred")
print(predictions)
print("Test")
print(y_test)
