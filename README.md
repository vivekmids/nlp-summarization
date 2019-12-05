## W266 Fall 2019 Final Project by Vivek Agarwal and Vivian Lu 

Slides can be accessed here: https://docs.google.com/presentation/d/1LTS-ZPZ5xNE7DFB-dNvVn-K2taopyWybQA2cc8NjUQk/edit?usp=sharing

## Abstract: 

Abstractive summarization--while still a challenging task in the NLP field--has primarily been conducted on the CNN/Daily Mail dataset. Due to the reporting style of most news articles, abstractive summarization for the CNN/Daily Mail dataset typically only requires the first paragraph of the entire article to perform relatively well. This paper seeks to perform a similar abstractive summarization (headlines) on the WikiHow dataset, a task that requires longer comprehension of multiple paragraphs.

# Main Contents of Repo 
1. **[Document name]** - Final Project Paper submitted 
2. **preparedata.py** - Preprocessing script for WikiHowSep.csv 
3. **w266_common** - A directory containing useful functions borrowed from W266 homework assignments for preprocessing. 
4. **forward** - A directory containing the training code for forward models, and the inference code for decoding sentences after training. 
5. **forward400** - Similar to item [4] above, but reflecting different parameters. 
6. **backward** A directory containing the training code for backward models, and the inference code for decoding sentences after training. 
7. **backwards20epochs** - Similar to item [5] above, but reflecting different parameters. 
8. **rmsprop** - A directory containing the training and inference code for forwards and backwards models but with RMSProp optimizers (items [4] through [6] all used Adam optimizers). 
9. **beam_search.py** - Code for implementing beam search 
10. **metrics.py** - Code borrowed from https://github.com/neural-dialogue-metrics/rouge to implement ROUGE calculations
11. **bleu_rouge_functions.py** - Code for calculating specific evaluation metrics 
12. **Error_Analysis.ipynb** - Code for analyzing errors in our abstractive summarizations. 

# Other contents in Repo worth tracking: 
    * Exploratory data analysis work was done in PreProcessing_wikihowsep.ipynb, PreProcessing_wikihowsep_vivek.ipynb
