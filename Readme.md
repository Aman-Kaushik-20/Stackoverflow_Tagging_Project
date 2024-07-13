#  Stack Overflow Tag Prediction Project

![5mkOX](https://github.com/user-attachments/assets/4c316b89-e159-4905-abe1-35ce3d8c8450)


This project is aimed at predicting the tags for questions posted on Stack Overflow. It involves preprocessing the data, preparing it for machine learning, and building a model to classify the questions based on their content.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Dependencies](#dependencies)
7. [License](#license)

## Features

- **Data Loading**: Loading and merging Stack Overflow questions and tags datasets.
- **Preprocessing**: Cleaning and preprocessing text data (questions, titles, and tags).
- **Feature Extraction**: Using TF-IDF for feature extraction.
- **Multi-Label Classification**: Training various classifiers to predict multiple tags for each question.
- **Evaluation**: Evaluating classifiers using metrics like precision, recall, F1-score, and Hamming loss.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/stackoverflow-tag-prediction.git
   cd stackoverflow-tag-prediction
   ```

2. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset

The dataset used for this project is the StackSample: 10% of Stack Overflow Q&A dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stacksample).

Place the `Questions.csv` and `Tags.csv` files in the project directory.

## Usage

1. **Run the preprocessing and training script**:
   ```sh
   python main.py
   ```

2. **Access the results**:
   - The script will output the performance metrics of various classifiers.

## File Structure

```
stackoverflow-tag-prediction/
│
├── data/
│   ├── Questions.csv        # Stack Overflow questions dataset
│   ├── Tags.csv             # Stack Overflow tags dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Preprocessing functions
│   ├── feature_extraction.py# Feature extraction functions
│   ├── training.py          # Model training and evaluation functions
│
├── main.py                  # Main script to run the project
├── requirements.txt         # List of Python packages required
└── README.md                # This README file
```

## Dependencies

- **numpy**: For numerical operations.
- **pandas**: For data manipulation.
- **nltk**: For natural language processing tasks.
- **beautifulsoup4**: For HTML parsing.
- **scikit-learn**: For machine learning tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Detailed Code Explanation

### Importing Required Libraries

```python
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import string
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score
```

### Loading the Dataset

```python
df_Ques = pd.read_csv('data/Questions.csv', encoding='latin')
df_Tags = pd.read_csv('data/Tags.csv', encoding='latin')

df_Ques.head()
df_Tags.head()
```

### Preprocessing Tags

```python
df_Tags['Tag'] = df_Tags['Tag'].astype(str)
grouped_tags = df_Tags.groupby('Id')['Tag'].apply(lambda tags: ' '.join(tags))
grp_tags = pd.DataFrame({'Id': grouped_tags.index, 'Tags': grouped_tags.values})

df_Ques.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
df = df_Ques.merge(grp_tags, on='Id')

df_new = df[df['Score'] > 5]
df_new['Tags'] = df_new['Tags'].apply(lambda x: x.split())
all_tags = [item for sublist in df_new['Tags'] for item in sublist]
keywords = nltk.FreqDist(all_tags)
word_freq = keywords.most_common(25)
tags_features = [word[0] for word in word_freq]

def most_common_tags(tags):
    return [tag for tag in tags if tag in tags_features]

df_new['Tags'] = df_new['Tags'].apply(lambda x: most_common_tags(x))
df_new['Tags'] = df_new['Tags'].apply(lambda x: x if len(x) > 0 else None)
df_new.dropna(subset='Tags', inplace=True)
```

### Preprocessing Text

```python
def clean_text(text):
    return text.lower().strip()

def clean_punct(text):
    for punct in string.punctuation:
        text = text.replace(punct, '')
    return text

token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatize_words(text):
    words = token.tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w, pos='v') for w in words])

def remove_stopwords(text):
    words = token.tokenize(text)
    return ' '.join([w for w in words if not w in stop_words])

df_new['Body'] = df_new['Body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
df_new['Body'] = df_new['Body'].apply(lambda x: clean_text(x))
df_new['Body'] = df_new['Body'].apply(lambda x: clean_punct(x))
df_new['Body'] = df_new['Body'].apply(lambda x: lemmatize_words(x))
df_new['Body'] = df_new['Body'].apply(lambda x: remove_stopwords(x))

df_new['Title'] = df_new['Title'].apply(lambda x: str(x))
df_new['Title'] = df_new['Title'].apply(lambda x: clean_text(x))
df_new['Title'] = df_new['Title'].apply(lambda x: clean_punct(x))
df_new['Title'] = df_new['Title'].apply(lambda x: lemmatize_words(x))
df_new['Title'] = df_new['Title'].apply(lambda x: remove_stopwords(x))

df_new['Combined_text'] = df_new['Title'] + ' ' + df_new['Body']
```

### Feature Extraction

```python
multilabel = MultiLabelBinarizer()
tfidf = TfidfVectorizer()

X1 = df_new['Body']
X2 = df_new['Title']
XC3 = df_new['Combined_text']
y = df_new['Tags']

y_ml = multilabel.fit_transform(y)

X1_tfidf = tfidf.fit_transform(X1)
X2_tfidf = tfidf.fit_transform(X2)
XC3_tfidf = tfidf.fit_transform(XC3)

X_train, X_test, y_train, y_test = train_test_split(XC3_tfidf, y_ml, test_size=0.2, random_state=0)
```

### Classification and Prediction

```python
classifiers = [SGDClassifier(), LogisticRegression(), MultinomialNB(), LinearSVC()]
prec_dict = {}
hamloss_dict = {}

for classifier in classifiers:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    ham = hamming_loss(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')

    clsname = classifier.__class__.__name__
    prec_dict[clsname] = prec
    hamloss_dict[clsname] = ham

    print(f'Classifier     : {clsname}')
    print(f'Hamming Loss   : {ham}')
    print(f'Precision Score: {prec}')
    print(f'Recall         : {recall_score(y_test, y_pred, average="weighted")}')
    print(f'F1-Score       : {f1_score(y_test, y_pred, average="weighted")}')
    print('\n\n')

x = ['how to write ml code in python and java i have data but do not know how to do it']
xt = tfidf.transform(x)
for classifier in classifiers[:2]:
    clf = OneVsRestClassifier(classifier)
    clf.predict(xt)
    print(multilabel.inverse_transform(clf.predict(xt)))
    print('\n\n')
```


