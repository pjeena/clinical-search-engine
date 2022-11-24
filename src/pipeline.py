
from utils import read_data
from preprocessing import output_text
from training import ModelTrain
from embeddings import return_embed
from top_n_searches import TopNrecommendations
#from top_n_searches import top_n
import pandas as pd
from gensim.models import Word2Vec

### Read the initial dataset ###
df = read_data("/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/input/Dimension-covid.csv")
df1 = read_data("/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/input/Dimension-covid.csv")  

### Pre-processing the data  ###
abstract_preprocessed  = output_text(df,"Abstract")

### Train the Skipgram and FastText models ###
model = ModelTrain(df,'Abstract', vector_size = 100,window_size=1)
skipgram = model.fit()
#fasttext = model_train(df,"Abstract","Fasttext",vector_size=100, window_size = 2)

### Loading our pretrained model ###
skipgram = Word2Vec.load('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/model_Skipgram.bin')
#FastText = Word2Vec.load('../output/model_Fasttext.bin')

### convert to columns vectors using skipgram ###
K1_abstract = return_embed(skipgram,df,"Abstract")
K1_abstract = pd.DataFrame(K1_abstract).transpose()    # Saving vectors of each abstract in data frame so that we can use directly while running code again
K1_abstract.to_csv('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/skipgram-vec-abstract.csv')

K1_title = return_embed(skipgram,df,"Title")
K1_title = pd.DataFrame(K1_title).transpose()  
K1_title.to_csv('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/skipgram-vec-title.csv')

### convert to columns vectors using FastText ###
#K2_abstract = return_embed(FastText,df,"Abstract")
#K2_abstract = pd.DataFrame(K1_abstract).transpose()    
#K2_abstract.to_csv('../output/FastText-vec-abstract.csv')

#K2_title = return_embed(FastText,df,"Title")
#K2_title = pd.DataFrame(K1_title).transpose()    
#K2_title.to_csv('../output/FastText-vec-title.csv')

### Loading our pretrained vectors of each abstract ###
K = read_data('/Users/piyush/Desktop/dsml_Portfolio/clinical_traits/final_project/output/skipgram-vec-abstract.csv')   
skipgram_vectors=[]                          #transforming dataframe into required array like structure as we did in above step
for i in range(df.shape[0]):
    skipgram_vectors.append(K[str(i)].values)

### Function to return 'n' similar results  ### 
query = 'lung'
model_name = 'Skipgram'
num_of_results = 5
top = TopNrecommendations(query,model_name,num_of_results)
Results = top.get_top_n_results()
