### Output
Output file processing is not implemented in this version of the system and is one of future tasks.  
According to rules of PoleVal task.  
The test data was sent in a json with format of:
```
[
   {
      "text":"Text paragraph",
      "id":"Id of the paragraph"
   }
]
```
The above json should be expanded by adding "answers" field which is a text format joined by a newline of:
```
type offset_start offset_end\t"text"
```
In an example:
```
[
   {
      "text":"There are no specifics - adds N. Napieraj",
      "id":"PCCwR-1.1-TXT/very_short/Dzienniki/1b.txt",
      "answers": "persName 30 41\tN. Napieraj\npersName_forename 30 32\tN.\npersName_surname 33 41\tNapieraj"
   }
]
```
File in above format can be parsed and tested with true labels and a script available at http://mozart.ipipan.waw.pl/~axw/poleval2018/