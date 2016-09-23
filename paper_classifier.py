import nltk
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import sentiment as VS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn import metrics
from paper_stats import *

#Set plot style to ggplot
plt.style.use('ggplot')

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
    print "Running models to predict excellent using stats only"
    y = df['excellent']
    X2 = get_stats(list(df['essay']))
    a = kfold(X2, y, model, K)
    print a
    #print "Running models to predict satisfactory..."
    #y = df['satisfactory']
    #X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    #X = pd.DataFrame(X.toarray())
    #b = kfold(X, y, model, K)
    #print b
    #print "Running models to predict good..."
    #y = df['good']
    #X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    #X = pd.DataFrame(X.toarray())
    #c = kfold(X, y, model, K)
    #print c
    print "Running models to predict excellent using vectorizer"
    y = df['excellent']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    d = kfold(X, y, model, K)
    print d

    print type(X)
    print X.shape
    print type(X2)
    print X2.shape

    print "Running models to predict excellent using both"
    M = pd.concat([X,X2], axis=1)
    #y = df['excellent']
    #e = kfold(M, y, model, K)

    #Skipping kfold above as slow!
    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')
    model.fit(M, y)

    print "Now performing feature selection to get better results"
    best = SelectFromModel(model, prefit=True,
                        threshold="mean")

    X_new = best.transform(M)
    X_new = pd.DataFrame(X_new)
    LR_2 = linear_model.LogisticRegression(C=10.0, penalty='l2',
                                         #regularization l1 since data are sparse
                                        class_weight='balanced')
    final = kfold(X_new, y, LR_2, 10)
    print final

def prediction(df, to_grade):
    data = pd.concat([df, to_grade])
    X_strings = data['essay']
    X = vectorizer.fit_transform(X_strings) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    X2 = get_stats(list(data['essay']))
    M = pd.concat([X,X2], axis=1)
    y = data['excellent']
    num2predict = - to_grade.shape[0]
    X_train, X_test = M.iloc[:-num2predict], M.iloc[-num2predict:]
    y_train, y_test = y.iloc[:-num2predict], y.iloc[-num2predict:]
    print "Running first model on training set"
    LR = linear_model.LogisticRegression(C=10.0, penalty='l1',
    class_weight='balanced').fit(X_train, y_train)
    print "Using trained model to predict for test set.."
    y_pred1 = LR.predict(X_test)
    print y_pred1
    print "Selecting best features and running second model..."
    model = SelectFromModel(LR, prefit=True,
                            #Threshold is the cut off for variable inclusion
                            #threshold="mean")
                            threshold="mean")
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    LR_2 = linear_model.LogisticRegression(C=10.0, penalty='l2',
                                         #regularization l1 since data are sparse
                                        class_weight='balanced')
    LR_2.fit(X_train_new, y_train)
    print "Making predictions based on tuned model..."
    y_pred2 = LR_2.predict(X_test_new)
    print y_pred2
    pickle.dump(y_pred1, open('y_pred1.p', 'wb'))
    pickle.dump(y_pred2, open('y_pred2.p', 'wb'))







if __name__ == '__main__':
    df = pickle.load(open('week4_model_table.p', 'rb'))
    print df.shape
    section1 = df[df.section == 202]
    section2 = df[df.section == 211]
    sections = pd.concat([section1, section2])
    to_grade = sections[sections.grade == 66]

    #Now filtering junk from training set
    df = df[df.essay != ''] #these conditions filter essays w/o content
    df = df[df.essay != ' ']
    df = df[df.grade != 70] #A mislabelled entry
    df = df[df.grade != 0] #Remove zero entries
    df = df[df.grade != 66] ##Remove ungraded (for now)
    print df.shape
    #df.grade.hist(bins=20)
    #plt.show()

    # Split into features and target
    K = 10 #10 folds

    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')


    #####model_iterator(df, model, K)
    #lr_kf = kfold(X, y, model, 10)
    #print lr_kf
    prediction(df, to_grade)
