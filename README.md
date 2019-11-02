## Rough brainstorm of plan: 

1. Preprocessing of wikihowsep.csv (found https://github.com/mahnazkoupaee/WikiHow-Dataset); status = DONE 
2. Working out input of sequences with word embeddings + padding + unknown vocabulary; status = In Progress
3. Baseline model of basic encoder (3 stacked LSTM) + decoder (3 stacked LSTM). Need to figure out attention layer and if beam search is necessary/time permits; status = Not yet started 
4. Train / figure out (3). This will need a .py file somewhere to save the model and also observe history of scores. 
5. Anticipated number of models we could do: 
* basic encoder-decoder with GlOve embeddings
* basic encoder-decoder with word2vec embeddings 
* basic encoder-decoder with Gl0ve embeddings + bidirectional 
* basic encoder-decoder with word2vec embeddings + bidirectional 

## Reference: 
1. Glove embeddings (https://nlp.stanford.edu/projects/glove/) - for now, using glove.42B.300d.zip
2. word2vec embeddings (?) 

## Current GPU set-up 
1. Vivian's GPU on GCP has tensorflow 1.14 on it. May update as needed 
2. IBM? 

## Paper: 
1. https://arxiv.org/pdf/1512.01712.pdf