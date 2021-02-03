import preprocessor as p
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textblob import TextBlob
from pathlib import Path
from langdetect import detect
import re
import glob
import csv


def preprocess_tweet(input_str):
    input_str = re.sub(r'b', '', input_str)
    input_str = p.clean(input_str)
    input_str = ' '.join(re.sub('\n', ' ', input_str).split())
    input_str = str(input_str).replace("\\n", "")
    input_str = ''.join([c for c in input_str if ord(c) < 128])
    input_str = re.sub(r'\\x[a-f0-9]{2,}', '', input_str)

    input_str = str(input_str)
    input_str = input_str.strip('\'"')
    input_str = input_str.replace(')', '').replace('(', '')
    input_str = input_str.replace('RT', '')
    input_str = input_str.replace(':', '')
    input_str = input_str.lower()
    stop_words = set(stopwords.words('english'))
    input_str = re.sub('[^a-z]', ' ', input_str)
    input_str = re.sub(' +', ' ', input_str)
    input_str = ' '.join([word for word in input_str.split() if word not in stopwords.words('english')])
    return input_str

def detect_lang(tweet):
    tweet = str(tweet)
    try:
        return detect(tweet)
    except:
        if len(tweet) > 2:
            return 'en'
        return 'unknown'

analyser = SentimentIntensityAnalyzer()
def sentiment_label(sentence):
    snt = analyser.polarity_scores(sentence)
    return ([snt['neg'], snt['neu'], snt['pos'], snt['compound']])


def vader_decide_category(l):
    l = l[:3]
    value = max(l)
    if l[0] == value:
        return 'negative'
    elif l[1] == value:
        return 'neutral'
    else:
        return 'positive'


def blob_decide_category(polarity):
    if polarity < 0:
        return 'negative'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'positive'


if __name__ == "__main__":
    dir_name = "/Users/danlinchen/Documents/CMPT825/project/data"
    all_files = glob.glob(dir_name + '/*.csv')
    for filename in all_files:
        print(filename)
        with open(filename, 'r') as csvfile:
            label_filename = Path(filename)
            output_file = 'label/' + label_filename.parts[-1].split('.')[0] + '-label.csv'
            print(output_file)
            with open(output_file, 'w') as fo:
                spamreader = csv.reader(csvfile)
                spamwriter = csv.writer(fo)
                for row in spamreader:
                    if len(row) == 3:
                        create_time = row[0]
                        text = row[1]
                        loc = row[2]
                    else:
                        try:
                            create_time = row[0]
                            text = row[1]
                            loc = ""
                        except:
                            print(row)
                    trimmed_text = preprocess_tweet(text)
                    lang = detect_lang(trimmed_text)

                    if lang == 'en':
                        vader_label = sentiment_label(trimmed_text)
                        blob_score = TextBlob(trimmed_text).sentiment.polarity

                        vader_category = vader_decide_category(vader_label)
                        blob_category = blob_decide_category(blob_score)
                        spamwriter.writerow([create_time,
                                             trimmed_text,
                                             vader_label,
                                             vader_category,
                                             blob_score,
                                             blob_category,
                                             loc])
