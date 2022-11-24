import numpy as np
import pandas as pd
import string        # used for preprocessing
import re            # used for preprocessing
import nltk          # the Natural Language Toolkit, used for preprocessing
import numpy as np   
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords       # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class Preprocessing():

    def __init__(self,text):
        self.text = text

    def get_preprocessed_data(self):
        new_text = self.text.lower()
        new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",new_text).split())
        new_text = re.sub(r'\d+', '', new_text)
        translator = str.maketrans('', '', string.punctuation)
        new_text = new_text.translate(translator)
        new_text = word_tokenize(new_text)
        stop_words = set(stopwords.words('english'))
        new_text = [i for i in new_text if not i in stop_words]
        lemmatizer = WordNetLemmatizer()
        new_text = [lemmatizer.lemmatize(token) for token in new_text]
        new_text = ' '.join(new_text)
        return new_text



def output_text(df,column_name): 
    for i in range(df.shape[0]):
        preprocess_text_class = Preprocessing(str(df[column_name][i]))
        df[column_name][i] = preprocess_text_class.get_preprocessed_data() 
    for text in df[column_name]:
        text = text.replace('\n',' ') 

    x = [word_tokenize(word) for word in df[column_name]]   #Tokenizing data for training purpose
    return x



def preprocessing_input(input_query):
    preprocess_text_class = Preprocessing(input_query)
    input_query = preprocess_text_class.get_preprocessed_data()
    input_query = input_query.replace('\n',' ')         
    return input_query  
