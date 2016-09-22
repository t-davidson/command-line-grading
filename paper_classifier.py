import nltk
import pickle
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import sentiment as VS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn import linear_model

##Defining stopwords list
stopwords = nltk.corpus.stopwords.words("english")
stemmer = nltk.stem.porter.PorterStemmer()

def tokenize(page):
    tokens = page.split()
    #Remove stopwords
    #tokens = [t for t in tokens if t not in stopwords]
    #stem tokens
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=tokenize,
    ngram_range=(1, 3),
    stop_words=stopwords,
    lowercase=True,
    use_idf=True,
    smooth_idf=True,
    norm='l2',
    decode_error='replace'
    )

def kfold(X, y, model, K):
    kf=StratifiedKFold(y, n_folds=K, random_state=42, shuffle=False)
    k=1
    precision = 0
    recall = 0
    accuracy = 0
    for train_index, test_index in kf:
        print "k = ", k
        #print "INDICES: ", train_index, test_index
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        print "Fitting model ", k
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p = metrics.precision_score(y_test, y_pred)
        r = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        print p, r, f1
        precision += p
        recall += r
        accuracy += f1
        k+=1
    return precision/K, recall/K, accuracy/K

def model_iterator(df, model, K):
    X_strings = df['essay']

    print "Running models to predict 0 grade (na)..."
    y = df['na']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    a = kfold(X, y, model, K)
    print a
    print "Running models to predict satisfactory..."
    y = df['satisfactory']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    b = kfold(X, y, model, K)
    print b
    print "Running models to predict good..."
    y = df['good']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    c = kfold(X, y, model, K)
    print c
    print "Running models to predict excellent..."
    y = df['excellent']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    d = kfold(X, y, model, K)
    print d

if __name__ == '__main__':
    df = pickle.load(open('week4_model_table.p', 'rb'))
    # Split into features and target
    K = 10 #10 folds

    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')
    model_iterator(df, model, K)
    #lr_kf = kfold(X, y, model, 10)
    #print lr_kf
