[10.06.2023]  
Removed features due to rewrite, to be implemented
1. Class weights
Since the model padding and outside should be assigned lower weights for training
2. IOB notation
Create available switch between IO and IOB outputs
3. Enable tree model instead of just concatenation
4. Re-implement CRF layer
Due to lack of support in Tensorflow-2.0 new CRF layer needs to be created from scratch
5. Original preprocessing
6. Testing module.
Due to complexity and reference on external python script (from creators of the dataset) testing module was removed.
To be re-added we need to:
* add parsing from prediction to text labels
* add validation on non-outside label
* refactor of the external script
