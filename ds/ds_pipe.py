import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


#constant
PATH_TFIDF = './tfidf.pickle'
PATH_MODEL = './kmeans.pickle'

#data pretrain
def text_without_url(text):
    return re.sub(r'http\S+', '', text)

def text_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    res = re.sub(r'[^\w\s]', ' ', text)
    return res

def remove_whitespace(text):
    return  " ".join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words('english') + stopwords.words('russian'))
    filtered_text = [word for word in text.split(' ') if word not in stop_words]
    return " ".join(filtered_text)

def tf_idf(text):
    with open(PATH_TFIDF, 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf.transform(text)

def define_class(text):
    with open('kmeans.pickle', 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans.predict(text)

def define_class_5(text):
    with open('kmeans_5.pickle', 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans.predict(text)

def main():
    # input data 
    log = 'immediate job 2eec8d41 fd97 4fa2 aa9f e2d30a26c150 execution failed trying enqueue job'

    # transform data
    log_clean = text_without_url(log)
    log_clean = text_lowercase(log_clean)
    log_clean = remove_punctuation(log_clean)
    log_clean = remove_whitespace(log_clean)
    log_clean = remove_stopwords(log_clean)

    log_df = pd.DataFrame(index=[0], columns=['log'])
    log_df['log'][0] = log_clean
    log_transform = tf_idf(log_df.log.values)

    # define class
    class_log = define_class(log_transform)[0]
    if class_log == 5:
        class_log = define_class_5(log_transform)[0]
    print(log)
    print('class -', class_log)

if __name__ == '__main__':
    main()