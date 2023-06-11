[10.06.2023]  
Sentence splitting was done during Part Of Speech tagging in an outside service.
If done separately inside this project the aligment between POS and new tokens would lead to unforseen errors.
Therefore, during the preprocessing we align sentence splitting to POS tags and not the other way.
