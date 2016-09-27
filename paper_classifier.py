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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
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

def kfold(X, y, model, K, multi_flag):
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
        if multi_flag is True:
            p = metrics.precision_score(y_test, y_pred, average='weighted')
            r = metrics.recall_score(y_test, y_pred, average='weighted')
            f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        else:
            p = metrics.precision_score(y_test, y_pred)
            r = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
        print p, r, f1
        precision += p
        recall += r
        accuracy += f1
        k+=1
    print "Precision: ", precision/K, " Recall: ", recall/K, " F1: ", accuracy/K
    return precision/K, recall/K, accuracy/K

def get_sections(section_ids):
    """Takes a list of section ids and returns a
    dataframe containing the students & essays
    that need to be graded"""
    sections = []
    for i in section_ids:
        df_section = df[df.section == i]
        sections.append(df_section)
    sections_df = pd.concat(sections)
    to_grade_df = sections_df[sections_df.grade == 66]
    print to_grade_df.shape[0], " essays to grade."
    return to_grade_df

def model_iterator(df, model, K):
    X_strings = df['essay']
    #print "Running models to predict excellent using stats only"
    y = df['excellent']
    X2 = get_stats(list(df['essay']))
    #a = kfold(X2, y, model, K)
    #print a
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
    #print "Running models to predict excellent using vectorizer"
    #y = df['excellent']
    X = vectorizer.fit_transform(X_strings, y) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    #d = kfold(X, y, model, K)
    #print d

    print "Running models to predict excellent using both"
    M = pd.concat([X,X2], axis=1)
    #y = df['excellent']
    #e = kfold(M, y, model, K)

    #Skipping kfold above as slow!
    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')

    P = OneVsRestClassifier(model)
    P.fit(M, y)

    print "Now performing feature selection to get better results"
    best = SelectFromModel(P, prefit=True,
                        threshold="mean")

    X_new = best.transform(M)
    X_new = pd.DataFrame(X_new)
    LR_2 = linear_model.LogisticRegression(C=10.0, penalty='l2',
                                         #regularization l1 since data are sparse
                                        class_weight='balanced')
    P = OneVsRestClassifier(LR_2)
    final = kfold(X_new, y, P, 10, True)
    print final

def prediction(df, to_grade, category_to_predict, week):
    data = pd.concat([df, to_grade])
    X_strings = data['essay']
    X = vectorizer.fit_transform(X_strings) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    X2 = get_stats(list(data['essay']))
    M = pd.concat([X,X2], axis=1)
    y = data[category_to_predict]
    num2predict = - to_grade.shape[0]
    X_test, X_train = M.iloc[:-num2predict], M.iloc[-num2predict:]
    print X_train.shape, X_test.shape
    y_test, y_train = y.iloc[:-num2predict], y.iloc[-num2predict:]
    print y_train.shape, y_test.shape
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
    #LR_2.fit(X_train_new, y_train)
    kfold(pd.DataFrame(X_train_new), y_train, LR_2, 10)
    print "Making predictions based on tuned model..."
    y_pred2 = LR_2.predict(X_test_new)
    print y_pred2
    pickle.dump(y_pred1, open('y_pred1.p', 'wb'))
    pickle.dump(y_pred2, open('y_pred2.p', 'wb'))
    output1 = 'p1_' + category_to_predict
    output2 = 'p2_' + category_to_predict
    to_grade[output1] = y_pred1
    to_grade[output2] = y_pred2
    graded = to_grade
    name = 'graded_'+category_to_predict+"_"+week
    pickle.dump(graded, open('graded_wk4.p', 'wb'))

def prediction_onevsrest(df, to_grade, category_to_predict, week):
    data = pd.concat([df, to_grade])
    X_strings = data['essay']
    X = vectorizer.fit_transform(X_strings) # fit vectorizer here
    X = pd.DataFrame(X.toarray())
    X2 = get_stats(list(data['essay']))
    M = pd.concat([X,X2], axis=1)
    y = data[category_to_predict]
    num2predict = - to_grade.shape[0]
    X_test, X_train = M.iloc[:-num2predict], M.iloc[-num2predict:]
    print X_train.shape, X_test.shape
    y_test, y_train = y.iloc[:-num2predict], y.iloc[-num2predict:]
    print y_train.shape, y_test.shape
    print "Running first model on training set"
    LR = linear_model.LogisticRegression(C=10.0, penalty='l1',
    class_weight='balanced')
    P = OneVsRestClassifier(LR)#.fit(X_train, y_train)
    print "Using trained model to predict for test set.."
    kfold(X_train, y_train, P, 10, True)
    y_pred = P.predict(X_test)
    print y_pred


if __name__ == '__main__':
    df = pickle.load(open('week4_model_table.p', 'rb'))
    SECTIONS = [202, 211] #My sections
    WEEK = 'week4' #Week to grade
    to_grade = get_sections(SECTIONS)

    #Now filtering junk from training set
    df = df[df.essay != ''] #these conditions filter essays w/o content
    df = df[df.essay != ' ']
    df = df[df.grade != 70] #A mislabelled entry
    df = df[df.grade != 0] #Remove zero entries
    df = df[df.grade != 66] ##Remove ungraded (for now)
    print df.shape
    df.grade.hist(bins=20)
    plt.show()

    # Split into features and target
    K = 10 #10 folds

    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')

    model_iterator(df, model, K)
    #lr_kf = kfold(X, y, model, 10)
    #print lr_kf
    ###PRED_CATEGORY = 'good'
    ####prediction(df, to_grade, PRED_CATEGORY , WEEK)
