### Model data processing
As model accepts only numerical values the data from text format needs to be processed accordingly:

Input:
Each token would have the following features in an input:
* tokens: mapped to embedding vector based on embedding vocabulary
* dependency_labels: mapped to one hot vector based on
* dependency_index: left as a number
* numerical_features: left as a list of numbers

Output: 
Each token in a sentence can have more than one label. 
Example:
"St John Street"
* St: geogName
* John: persName_forename, persName, geogName
* Street: geogName
In model output this would map to:
[geogName, persName-persName_forename-geogName,geogName]

Afterwards we create target vocabulary based on concatenated labels and the output is mapped to one hot vector according to those target indexes. 
