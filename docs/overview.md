### Hierarchical NER
The task of this module is to train a model that can recognize hierarchical Named Entities.
In most common approach each named entity can be part of only of one type.

However, in reality the names are more complex. In phrase:
```commandline
St John Street
```
St John Street is a place, but John is a person.   
This projects aims to detect this complexity by training Sequence-to-Sequence model.

This module includes:
1) Preprocessing the data for training
   * load the data and embeddings
   * alignment between dependency parsing, tokenization and targets
   * add additional features
   * processing data from to vector format
2) Training the model - BidirectionalRNN
3) Validation