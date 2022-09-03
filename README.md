# README
Codes and data for the Coling2022 paper:   
AiM:Taking Answers in Mind to Correct Chinese Cloze Tests in Educational Applications

The datasets are available now and codes will come soon.  

The EinkCC and SynCC dataset are [available](https://drive.google.com/file/d/1NsXHEWbUCHt8uGUcWgk309fuUPdzLlAH/view?usp=sharing).  
The meanings of the fields in the json file are as follows:  
`1.file: the image path`  
`2.handwriting: the hand-written content in the image`  
`3.answer: the true answer for the question`  
`4.bin_label(if exists): the binary sequence label for the answer, which indicates whether the token in the true answer appears in the hand-written content. Notice that the binary sequence labels are not used in practice.`
`5.edit_label(if exists): the edit sequence label for the answer. For details please refer to the paper.`  
`6.label: The label indicates whether the hand-written content is a correct anwser(1 for correct and 2 for wrong)`  

For the HWCC, you can get the HWDB2.x benchmark first. Then you can generate HWCC dataset for the CCC task through the function `random_extend_data` in the `random_generate_neg_data.py` script. 
