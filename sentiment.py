from nltk.corpus import twitter_samples, stopwords;
from nltk.tag import pos_tag;
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
import re, string
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

custom_tweet = "I am in love with the food in this place."

stop_words = stopwords.words('english')
positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")
text = twitter_samples.strings('tweets.20150430-223406.json')
positive_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')

def remove_noise(tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleans_tokens_list):
    for tweet_tokens in cleans_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_cleaned_tweets_tokens_list = []
negative_cleaned_tweets_tokens_list = []

for tokens in positive_tweets_tokens:
    positive_cleaned_tweets_tokens_list.append(remove_noise(tokens, stop_words))
for tokens in negative_tweets_tokens:
    negative_cleaned_tweets_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tweets_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)
all_neg_words = get_all_words(negative_cleaned_tweets_tokens_list)
freq_dist_neg = FreqDist(all_neg_words)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tweets_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tweets_tokens_list)

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
custom_tweet = input("Say something: ")
custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))