# A basic sentiment analysis to check polarity (positive/negative) of a sentence using Python
This program is a basic sentiment analysis script capable of checking the polarity (positive/ negative) of a sentence. It uses the nltk library and it's functions to achieve the goal. The data used to train the model is from the twitter samples within the library. And, the model is trained using these data after normalization (lemmatization) and a simple noise cleaning process with the help of a Naive Bayes Classifier.
The full walkthrough of the implementation is available [a link](https://github.com/user/repo/blob/branch/other_file.md)here.
# How to run
  1. Clone the repository
  2. pip install nltk
  3. open python shell
  4. import nltk
  5. nltk.download('twitter_samples')
  6. nltk.download('punkt')
  7. nltk.download('wordnet')
  8. nltk.download('averaged_perceptron_tagger')
  9. nltk.download('stopwords')
  10. Run the script
