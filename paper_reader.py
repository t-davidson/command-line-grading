from os import listdir
from os.path import isfile, join
from unidecode import unidecode
import PyPDF2
import docx
import pandas as pd
import numpy as np
import copy
import string
import re
import pickle
PATH = '../data/'
WEEK = 'week2'
MY_STUDENTS = pd.read_csv('../data/intro_students.csv')
MY_STUDENTS = list(MY_STUDENTS.net_id)

def get_files(path,week):
    path = path+week
    files = [f for f in listdir(path) if isfile(join(path, f))]
    pdfs = [f for f in files if f.endswith('.pdf')]
    docs = [f for f in files if f.endswith('.docx')]
    txts = [f for f in files if f.endswith('.txt')]
    return pdfs, docs, txts

def parse_pdf(file_path):
    file = open(file_path, 'rb')
    reader = PyPDF2.PdfFileReader(file)
    try:
        page_0 = reader.getPage(0)
        text = page_0.extractText()
        try:
            page_1 = reader.getPage(1)
            text = text + ' ' + page_1.extractText()
        except IndexError:
            pass
    except IndexError:
        print "Error found, cannot get first page"
    #Now to decode the unicode elements
    decoded = unidecode(text) #decode unicode
    decoded2 = decoded.replace('\nO', '') #replace tab chars
    decoded3 = decoded2.replace('\n', '') #replace newline chars
    text = decoded3
    return text

def parse_docx(file_path):
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    text =  '\n'.join(fullText)
    text = unidecode(text)
    text = text.encode('utf-8')
    text = text.strip()
    return str(text)

def find_txts(files, net_ids):
    """Most of the .txt files are just additional info (one created for
    each submission). This script finds the text files that contain a net ID
    that was not linked to either a .pdf or a .docx. These files are
    student submissions in .txt format.
    Returns: a list of text files + a list of net IDs for students w/o
    anything submitted"""
    pdfs = files[0]
    docs = files[1]
    txts = files[2]
    missing = copy.deepcopy(net_ids)
    valid_txts = []
    for p in pdfs:
        for nid in net_ids:
            if nid in p:
                missing.remove(nid)
    for d in docs:
        for nid in net_ids:
            if nid in d:
                missing.remove(nid)
    #Now go through remaining missing and
    #Add txts to new list and remove from missing
    for t in txts:
        for m in missing:
            if m in t:
                missing.remove(m)
                valid_txts.append(t)
    return valid_txts, missing

def parse_txt(file_path):
    f = open(file_path)
    text = f.read()
    text1 = text.split("Submission Field:")
    text2 = text1[1].split("Comments:")
    submission = text2[0]
    text4 = text2[1]
    text5 = text4.split("Files:")
    comments = text5[0]
    if len(submission) > len(comments):
        submission = re.sub('<[^<]+?>', '', submission)
        return submission
    else:
        comments = re.sub('<[^<]+?>', '', comments)
        return comments

def get_text(PATH, WEEK, files, MY_STUDENTS):
    pdfs = files[0]
    docs = files[1]
    txts = files[2]
    output = {}
    for p in pdfs:
        for student in MY_STUDENTS:
            if student in p:
                path = PATH+WEEK+'/'+p
                text = parse_pdf(path)
                output[student] = text
    for d in docs:
        for student in MY_STUDENTS:
            if student in d:
                path = PATH+WEEK+'/'+d
                text = parse_docx(path)
                output[student] = text
    txts = find_txts(files, MY_STUDENTS)
    for t in txts[0]:
        for student in MY_STUDENTS:
            if student in t:
                path = PATH+WEEK+'/'+t
                text = parse_txt(path)
                output[student] = text
    missing = txts[1]
    for m in missing:
        output[m] = "NO ASSIGNMENT SUBMITTED"
    return output

def grader(df):
    grade_list = []
    num_remaining = df.shape[0]
    for row in df.iterrows():
        print row[1]['papers']
        grade = raw_input("Enter a grade for the student [80,90,100]: ")
        grade_list.append(grade)
        num_remaining-=1
        print '\n'
        print str(num_remaining)+" students waiting to be graded."
        print '\n'
    return grade_list

def grade_change(df, net_id):
    """Opens the pickle file of grades and allows the
    user to manually change the grade for a given student,
    specified by their net id in the function call.

    Returns a pandas dataframe of grades"""

if __name__ == '__main__':
    files = get_files(PATH, WEEK)
    texts = get_text(PATH, WEEK, files, MY_STUDENTS)
    net_ids = texts.keys()
    papers = texts.values()
    df = pd.DataFrame()
    df['net_ids'] = net_ids
    df['papers'] = papers
    #pickle.dump( df, open( "week1_papers.p", "wb" ) )
    #df['grades'] = 70 #assign base value to all
    grade_list = grader(df)
    print grade_list
    df['grades'] = grade_list
    name = WEEK + "_graded.p"
    pickle.dump( df, open( name, "wb" ) )
    print df['grades']
