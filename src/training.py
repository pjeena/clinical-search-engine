import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from preprocessing import output_text


class ModelTrain():

    def __init__(self,dataframe,column_name,vector_size,window_size):
        self.dataframe = dataframe
        self.column_name = column_name
        self.vector_size = vector_size
        self.window_size = window_size

    def fit(self):
        x = output_text(self.dataframe,self.column_name)
        skipgram = Word2Vec(x, vector_size =self.vector_size, window = self.window_size, min_count=2,sg = 1)
        skipgram.save('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/model_Skipgram.bin')
        return skipgram


