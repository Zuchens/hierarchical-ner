### Train data
"data/input_data" holds input files.
train dataset:
subset from NKJP  "National Corpus of Polish"  
(Available at: http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=view&target=NKJP-PodkorpusMilionowy-1.2.tar.gz [11.06.2023])  
Data sample available at: data_example.json  
Data format: JSON  

Note: 
Data explained above already have some preprocessing steps done such as dependency parsing

```
Each element in text list consist of one paragraph from train daataset.
the paragraph element has following fields:
* offsets2Entities:  mapping between offset of each token in the paragraph and their named entity (see <NAMED_ENTITY>).
* text: full paragraph text
* tokens: paragraph text separated to tokens (not separated by sentences)
* dependencies: index of the dependency tree (separated by sentences)
* dependencyLabels: labels of the dependency tree  (separated by sentences)
```

```
<NAMED_ENTITY>
Labels for each named entity
Each word can be a part of one or multiple Named Entity Phrases.
Each Named Entity Phrases can consist of one or multiple tokens.
Each named entity consist of:
* type: named entity type,
* subtype: <Optional> named entity subtype,
* text: named entity text,
* offsets: list of tokens of this Named Entit Phrase, described by their start and finish ({"to": int, "from": int})
```

