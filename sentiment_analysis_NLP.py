
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting=3)

"""
import re
review = re.sub('[^a-zA-Z, ]','', dataset['Review'][0])
review = review.lower()

import nltk
nltk.download('stopwords')

review = review.split()
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    