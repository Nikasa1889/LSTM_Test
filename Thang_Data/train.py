import logging
import argparse
import os
import json
import csv
import pandas as pd
from nltk.stem import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


def load_id_list(emotion, idlist, train):
    idListFilename = emotion
    if train:
        idListFilename = idListFilename + "_train_uniq.txt"
    else:
        idListFilename = idListFilename + "_test_uniq.txt"
    idListFilePath = os.path.join(idlist, idListFilename)
    fidlist = open(idListFilePath, 'r')
    ids = []
    for line in fidlist:
        ids.append(line[:-1])
    fidlist.close()
    return ids


def load_data(emotions, input, idlist, train):
    data = []
    target = []
    index = 0 
    # Tokenizer
    tok = lambda x: TreebankWordTokenizer().tokenize(x)
    # Stemmer
    stemmer = PorterStemmer()
    ste = lambda x: stemmer.stem(x)
    stemmeta = lambda x: map(ste,x)
    # Joiner
    jo = lambda x: " ".join(x)

    for emotion in emotions:
        inputFilename = "processed_" + emotion + ".csv"
        inputFilePath = os.path.join(input,inputFilename)
        ids = load_id_list(emotion, idlist,train)
        df = pd.read_csv(inputFilePath,  names=['id','name','url','ups','downs','created_utc','title','permalink','subreddit','num_comments','author','author_link_karma','author_comment_karma','author_created_utc'])
        titles = df[df['name'].isin(ids)]['title'].map(tok).map(stemmeta).map(jo).tolist()
        # tts = df[df['name'].isin(ids)]['title'].tolist()
        # titles = []
        # for title in tts:
        #     try:
        #         tokens = TreebankWordTokenizer().tokenize(title)
        #         tokens = [stemmer.stem(token) for token in tokens]
        #         tokenized_title = " ".join(tokens)
        #         titles.append(tokenized_title)
        #     except Exception:
        #         print title
        classes = [index]*len(titles)
        data = data + titles
        target = target + classes
        index = index + 1
    text = {'data': data, 'target': target }
    return text

def train(text, ngram, c, loss):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, use_idf=True,ngram_range=(1,ngram),
                                 stop_words='english')
    trainText = vectorizer.fit_transform(text['data'])
    classifier = LinearSVC(loss=loss, penalty='l2',dual=False, tol=1e-3, C=c)
    classifier.fit(trainText,text['target'])
    return classifier, vectorizer

def test(classifier,vectorizer, text):
    testText = vectorizer.transform(text['data'])
    pred = classifier.predict(testText)
    score = metrics.accuracy_score(text['target'], pred)
    print("accuracy:   %0.3f" % score)

def main():
    parser = argparse.ArgumentParser(description='Download the whole subreddit')
    parser.add_argument("--emotions",'-e', dest="emotions", nargs='+', required=True, help="List of the emotions")
    parser.add_argument('--input', '-in', type=str, dest='input', help='Path to emotion data folder', required=True)
    parser.add_argument('--idlist', '-id', type=str, dest='idlist', help='Path to image id list folder', required=True)
    parser.add_argument('--ngram', '-ng', type=int, dest='ngrams', default=1, help='Maximum number of ngrams', required=False)
    parser.add_argument('--para_c', '-c', type=float, dest='c', default=1, help='Parameter C of liblinear', required=False)
    parser.add_argument('--para_loss', '-loss', type=str, dest='loss', default='squared_hinge', help='Loss function', required=False)
    # python train.py -e happy creepy rage gore -in /mnt/d/PhD/Semester1/rharvest/processed_no_text/ -id /mnt/d/PhD/Semester1/rharvest/processed_no_text/ -ng 2 -c 1 -loss squared_hinge
    args = parser.parse_args()
    trainData = load_data(args.emotions,args.input,args.idlist,True)
    print("Training data loaded")
    testData = load_data(args.emotions,args.input,args.idlist,False)
    print("Testing data loaded")
    classifier, vectorizer = train(trainData, args.ngrams, args.c,args.loss)
    print("Trained with training data")
    test(classifier,vectorizer, testData)


if __name__ == "__main__":
    main()