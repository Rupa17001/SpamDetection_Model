
import re
from nltk.stem import WordNetLemmatizer 
import pandas as pd
from nltk.corpus import stopwords

# variable 
lemmatizer = WordNetLemmatizer()
# with pandas library reading csv file that is separated on the basis of tab and as the data is unstructures the columns have been given names "label", "message"
messages = pd.read_csv(r'C:\Users\Admin\Desktop\nlp\smsspamcollection\SMSSpamCollection.txt', sep='\t',
                       names=["label", "message"])

# Corpus is the blank list
corpus = []

# the extra space and words that has no meaning are removed here using regex , stopwords anf lemmatization (just to reduce text)
# then the sorted text is added in corpus[] list
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# text inside the corpus is being converted into vector
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=2500)
X = tf.fit_transform(corpus).toarray()
print (X)

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
print (y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

# print (y_pred)