'''
@author: Julia Biswas
'''

import numpy as np
import pandas as pd
from bayesia_sample_functions import clean_data, train

def main():
    data = pd.read_csv('dating data.csv', engine='python')
    data = clean_data(data)
    
    data.to_csv('cleaned_data.csv', index=False)
    
    train_data = data.iloc[:, :int(.8*len(data))] #80% of the data for training
    P_Y, P_Xi_Y = train(train_data)
    
    with open('bayesia.npy', 'wb') as f:
        np.save(f, P_Y)
        np.save(f, P_Xi_Y)

if __name__ == "__main__":
    main()
    
    
    
    
    


