"""File to test out neural embeddings"""
import pickle
import numpy as np
import pandas as pd
import string
from gensim import utils
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from random import seed
from random import shuffle
from paper_classifier import *
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2

#Tag doc2vec.TaggedDocument(bow, [count])
#Access model.docvecs[count]

def clean_text(text):
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return text.lower()

def doc_iterator(df):
    """Parses text documents from the essay field of the
    dataframe, cleans text, tokenizes, and returns it
    as an iterator"""
    for i in range(0, df.shape[0]):
        yield clean_text(df.essay.iloc[i]).split()

def tagged_iterator(text_iterator):
    """Processes texts in the doc_iterator and returns
    an iterator of tagged documents"""
    count=0
    for bow in text_iterator:
        if len(bow) > 0:
            yield doc2vec.TaggedDocument(bow, [count])
            count += 1
    print count-1

def docs_shuffle(iterator):
    """Shuffles the iterator"""
    list_of_docs = []
    for i in iterator:
        list_of_docs.append(i)
    shuffle(list_of_docs)
    for d in list_of_docs:
        yield d

def build_X(df, model, size):
    X = np.zeros((df.shape[0], size))
    for i in range(0, df.shape[0]):
        col = model.docvecs[i]
        X[i] = col
    return X

if __name__ in '__main__':
    df = pickle.load(open('week4_model_table.p', 'rb'))
    df = df[df.essay != ''] #these conditions filter essays w/o content
    df = df[df.essay != ' ']
    df = df[df.grade != 70] #A mislabelled entry
    df = df[df.grade != 0] #Remove zero entries
    df = df[df.grade != 66] ##Remove ungraded (for now)
    print df.shape

    docs = doc_iterator(df)
    tagged = tagged_iterator(docs)
    #tagged = docs_shuffle(tagged) #shuffle order of tagged

    size=10000
    ####Odd, when I don't do feature selection I get junk if size > 100
    ###but when I do feature selection I get better results with larger size
    model = doc2vec.Doc2Vec(
        tagged,
        size=size,
        min_count=3,
        workers=4,
        iter=20,
    )

    #for epoch in range(100): ###This appears to make no difference
    #    seed(randint(0,100))
    #    tagged = docs_shuffle(tagged)
    #    model.train(tagged)
    #print model.most_similar('unequal')
    model.save('doc2vecmodel')
    #print model.docvecs[767]
    #model.build_vocab(tagged) #I think my code does this by including tagged in model spec

    #num_docs = df.shape[0]
    X = build_X(df, model, size)
    y = df.grade

    ###Performs badly with high dim vector
    model = linear_model.LogisticRegression(C=10.0, penalty='l1',
                                            class_weight='balanced')

    P = OneVsRestClassifier(model)
    kfold(pd.DataFrame(X), y, P, 5, True)
    y_pred = P.predict(X)
    #print y_pred

    #Single class, excellent
    y = df.excellent
    kfold(pd.DataFrame(X), y, model, 5, False)
    y_pred = model.predict(X)
    #print y_pred

    SFM = SelectFromModel(model, prefit=True,threshold="10*mean")
    X_new = SFM.transform(X)
    print X_new

    print pd.DataFrame(X_new).shape
    model2 = linear_model.LogisticRegression(C=10.0, penalty='l2',
                                         #regularization l1 since data are sparse
                                        class_weight='balanced')

    kfold(pd.DataFrame(X_new), y, model2, 5, False)
    y_pred = model2.predict(X_new)
    print y_pred
