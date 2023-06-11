### Model  
Current implementation of the model consists of following elements:
On input the model processes each sentence, split by tokens and predicts what output label would be 
* Embedding input: Input layer to embedding lookup table, 
* Features input: Input layer of additional features for each token. Those can include for example information such as: is token numerical or uppercase, what it's is dependency label or index.
* Bidirectional RNN
* Time distributed output: Output layer where for each token is mapped to model label.


```
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 embedding_input (InputLayer)   [(None, 300)]        0           []                               
                                                                                                  
 embedding (Embedding)          (None, 300, 300)     26589300    ['embedding_input[0][0]']        
                                                                                                  
 words (InputLayer)             [(None, 300, 30)]    0           []                               
                                                                                                  
 concatenate (Concatenate)      (None, 300, 330)     0           ['embedding[0][0]',              
                                                                  'words[0][0]']                  
                                                                                                  
 masking (Masking)              (None, 300, 330)     0           ['concatenate[0][0]']            
                                                                                                  
 bidirectional (Bidirectional)  (None, 300, 600)     1137600     ['masking[0][0]']                
                                                                                                  
 dropout (Dropout)              (None, 300, 600)     0           ['bidirectional[0][0]']          
                                                                                                  
 time_distributed (TimeDistribu  (None, 300, 600)    360600      ['dropout[0][0]']                
 ted)                                                                                             
                                                                                                  
 masking_1 (Masking)            (None, 300, 600)     0           ['time_distributed[0][0]']       
                                                                                                  
 dense_1 (Dense)                (None, 300, 18)      10818       ['masking_1[0][0]']              
                                                                                                  
==================================================================================================
```