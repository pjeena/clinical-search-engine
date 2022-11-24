#defining function to define cosine similarity
#from numpy import dot
import numpy as np
#from numpy.linalg import norm
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd
from utils import read_data
from embeddings import get_mean_vector
from preprocessing import preprocessing_input
from sklearn.metrics.pairwise import cosine_similarity



class TopNrecommendations():

    def __init__(self,query,model_name,num_of_results):
        self.query = query
        self.model_name = model_name
        self.num_of_results = num_of_results

    def get_top_n_results(self):
        df = read_data("/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/input/Dimension-covid.csv")
        aa = np.array([1,2,3])
        word2vec_model = Word2Vec.load('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/model_Skipgram.bin')
        K=pd.read_csv('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/skipgram-vec-abstract.csv')

        input_query = preprocessing_input(self.query)
    
        input_query_vector=get_mean_vector(word2vec_model,input_query)
        p=[]                          #transforming dataframe into required array like structure as we did in above step
        for i in range(df.shape[0]):
            p.append(K[str(i)].values)    
        x=[]        #Converting cosine similarities of overall data set with input queries into LIST
        for i in range(len(p)):
            x.append(cosine_similarity(input_query_vector.reshape(1, -1),p[i].reshape(1, -1)))

       #index_top_n = np.argsort(np.array(x).flatten())
        index_top_n = np.flip(np.argsort(np.array(x).flatten()))[0:self.num_of_results]
        sim_scores = np.array(x).flatten()[index_top_n]
 #       df = read_data("/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/input/dimensions-covid19-export-2021-09-01-h15-01-02_clinical_trials.csv")
        return df.loc[index_top_n, ['Trial ID', 'Title','Abstract','Publication date']],sim_scores     #returning dataframe (only id,title,abstract ,publication date)

        