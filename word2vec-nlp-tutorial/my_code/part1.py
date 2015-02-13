import os
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.ensemble import RandomForestClassifier as rf

stop = stopwords.words("english")

def reviews2words(raw_review):
    review_text = bs(raw_review).get_text()
    letters_only = re.sub("[^a-zA-z]", " ", review_text)
    wordlist = letters_only.lower().split()
    words = [w for w in wordlist if not w in stop]
    return " ".join(words)

if __name__ == "__main__":
    train = pd.read_csv(".\data\labeledTrainData.tsv", header = 0, delimiter = "\t", quoting=3)
    clean_train = []
    print 'Start cleaning reviews...\n'
    for i in xrange(0, train['review'].size):
        clean_train.append(reviews2words(train['review'][i]))
        
    print 'Creating the bag of words...\n'
    vectorizer = cv(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    train_data_features = vectorizer.fit_transform(clean_train)
    train_data_features = train_data_features.toarray()
    # print train_data_features.shape
     
    vocab = vectorizer.get_feature_names()
    # print vocab
    
    forest = rf(n_estimators = 100)
    forest = forest.fit(train_data_features, train['sentiment'])

    test = pd.read_csv(".\data\\testData.tsv", header=0, delimiter="\t", quoting=3)
    # Create an empty list and append the clean reviews one by one

    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,num_reviews):
        if((i + 1) % 1000 == 0):
            print "Review %d of %d\n" % (i + 1, num_reviews)
        clean_review = reviews2words(test["review"][i])
        clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)