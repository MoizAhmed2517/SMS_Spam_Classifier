import pandas as pd
import numpy as np
from nltk import accuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter

ps = PorterStemmer()

data = pd.read_csv('E:\Projects\ML projects\SMS_Spam_Classifier\spamdata.csv', encoding='ISO-8859-1')

# Removing all null colummns and values
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, axis=1)

# Renaming our columns name for better understanding
data.rename({'v1': 'target', 'v2': 'text'}, inplace=True, axis=1)

# Label encider for classifying our target
data['target_Binary'] = LabelEncoder().fit_transform(data['target'])

# Again checking for missing values
data.isnull().sum()

# checking for duplicate values
data = data.drop_duplicates(keep='first')

# EDA - Exploratory Data Analysis

# ----- Checking for the percentage of ham and Spam -----
# spamVal = data['target'] == 'spam'
# print(np.round((spamVal.sum() / data.shape[0]) * 100, 2))
# hamVal = data['target'] == 'ham'
# print(np.round((hamVal.sum() / data.shape[0]) * 100, 2))

# ----- Another way to do the same thing -----

valuesCount = data['target'].value_counts()
# plt.pie(valuesCount, labels=['ham', 'spam'], explode=[0.2, 0], startangle=180, shadow=True, autopct='%1.1f%%')
# plt.legend()
# plt.show()

# Counting number of characters in each text segment
data['num_char'] = data['text'].apply(len)

# counting number of words in each text segment
data['num_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# Counting number of sentences in each text segment
data['num_sent'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# checking the statistics

# print(data.describe().transpose())

# Performing statical interference between ham and spam messages
# print(data[data['target_Binary'] == 0][['num_char', 'num_words', 'num_sent']].describe())
# print('\n' + 'Spam _____')
# print(data[data['target_Binary'] == 1][['num_char', 'num_words', 'num_sent']].describe())

# Using seaborn for better visualization
# plt.figure(figsize=(12, 12))


# sns.histplot(data[data['target_Binary'] == 0]['num_char'])
# sns.histplot(data[data['target_Binary'] == 1]['num_char'], color='red')


# Observing the relation between number of words, characters, sentences
# sns.pairplot(data, hue='target_Binary')
# # sns.heatmap(data.corr(), annot=True)
# plt.show()

# function that will be used to validate the text data
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    l = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            l.append(ps.stem(i))
    return " ".join(l)


data['transform_text'] = data['text'].apply(transform_text)
# # print(type(data.transform_text.iloc[:]))
#
# # Making word cloud
# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
# spam_wc = wc.generate(data[data['target_Binary'] == 1]['transform_text'].str.cat(sep=" "))
# ham_wc = wc.generate(data[data['target_Binary'] == 0]['transform_text'].str.cat(sep=" "))
# plt.figure(figsize=(12, 12))
# plt.imshow(spam_wc)
# plt.imshow(ham_wc)
# plt.show()

# checking for the top 30 and 50 words in each category

# spam_corpus = []
# for msg in data[data['target_Binary'] == 1]['transform_text'].tolist():
#     for word in msg.split(' '):
#         spam_corpus.append(word)
#
# sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()

# cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

GNB = GaussianNB()
MNB = MultinomialNB()
BNB = BernoulliNB()
# X = cv.fit_transform(data['transform_text']).toarray()
y = data['target_Binary'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)

# fitting GaussianNB model

# GNB.fit(X_train, y_train)
# y_pred1 = GNB.predict(X_test)
# print(accuracy_score(y_test, y_pred1))
# print(confusion_matrix(y_test, y_pred1))
# print(precision_score(y_test, y_pred1))
#
#
# # fitting GaussianNB model
# print('\n' + '___MNB___')
# MNB.fit(X_train, y_train)
# y_pred2 = MNB.predict(X_test)
# print(accuracy_score(y_test, y_pred2))
# print(confusion_matrix(y_test, y_pred2))
# print(precision_score(y_test, y_pred2))
#
# print('\n' + '___BNB___')
# BNB.fit(X_train, y_train)
# y_pred3 = BNB.predict(X_test)
# print(accuracy_score(y_test, y_pred3))
# print(confusion_matrix(y_test, y_pred3))
# print(precision_score(y_test, y_pred3))

# Usinf TFIDF vectorizer with GNB, MNB, BNB

X_1 = tfidf.fit_transform(data['transform_text']).toarray()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1, y, test_size=0.2, random_state=True)

# fitting GaussianNB model

print('\n' + '___GNB_t___')
GNB.fit(X_train1, y_train1)
y_pred4 = GNB.predict(X_test1)
print(accuracy_score(y_test1, y_pred4))
print(confusion_matrix(y_test1, y_pred4))
print(precision_score(y_test1, y_pred4))


# fitting GaussianNB model
print('\n' + '___MNB_t___')
MNB.fit(X_train1, y_train1)
y_pred5 = MNB.predict(X_test1)
print(accuracy_score(y_test1, y_pred5))
print(confusion_matrix(y_test1, y_pred5))
print(precision_score(y_test1, y_pred5))

print('\n' + '___BNB_t___')
BNB.fit(X_train1, y_train1)
y_pred6 = BNB.predict(X_test1)
print(accuracy_score(y_test1, y_pred6))
print(confusion_matrix(y_test1, y_pred6))
print(precision_score(y_test1, y_pred6))

# Finalizing  that TFIDF vectorize and MNB at max_features of 3000 gives best prediction and no effect can be seen with scaling so removing it.