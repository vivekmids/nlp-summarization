############
##### helpful functions to help with formatting when printing: 
############
def printmd(string):
    display(Markdown(string)) #just a pretty print 

############
##### function for BLEU 
#############

from nltk.translate.bleu_score import sentence_bleu
# reference: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

def calc_indiv_BLEU(id_text, text_df, headline_df): 
    # This function will take the following as inputs: 
    # id_text: the index you are interested in 
    # gen_text_df: the sequences that hold the full text 
    # headline_df: the sequences that hold the headline text 
    
    # -- Step 1: generate the decoded sequence from a given sample of text
    gen_output = decode_sequence(text_df[id_text].reshape(1,-1))
    split_output = gen_output.split(" ")
    candidate = [item for item in split_output if (item!="." and item!="")] #get rid of empty spaces and periods 
    # -- Step 2: generate the true headline summary from our labelled headline text
    gen_ref = seq2summary(headline_df[id_text])
    split_ref = gen_ref.split(" ")
    #get rid of empty spaces and periods (there shouldn't be any as we already cleaned the headline, but just in case)
    reference = [item for item in split_ref if (item!="." and item!="")] 
    # -- Step 3: calculate BLEU 
    score = sentence_bleu(gen_ref, gen_output, weights=(1, 0, 0, 0))
    # we can alternate weights for cumulative scores afterwards 
    # For now, BLEU is based on unigram counts 
    return(score)

###############
##### function for rouge 
############### 
# PULL METRICS.PY FILE!! 
from metrics import rouge_n_sentence_level
# metrics.py is taken from https://github.com/neural-dialogue-metrics/rouge

# Other useful links to keep in mind: 
# https://stackoverflow.com/questions/38045290/text-summarization-evaluation-bleu-vs-rouge

def calc_indiv_rouge(id_text, text_df, headline_df, rouge_n): 
    # This function will take the following as inputs: 
    # id_text: the index you are interested in 
    # gen_text_df: the sequences that hold the full text 
    # headline_df: the sequences that hold the headline text 
    
    # -- Step 1: generate the decoded sequence from a given sample of text
    gen_output = decode_sequence(text_df[id_text].reshape(1,-1))
    split_output = gen_output.split(" ")
    candidate = [item for item in split_output if (item!="." and item!="")] #get rid of empty spaces and periods 
    # -- Step 2: generate the true headline summary from our labelled headline text
    gen_ref = seq2summary(headline_df[id_text])
    split_ref = gen_ref.split(" ")
    #get rid of empty spaces and periods (there shouldn't be any as we already cleaned the headline, but just in case)
    reference = [item for item in split_ref if (item!="." and item!="")] 
    # -- Step 3: calculate rouge
    recall, precision, rouge = rouge_n_sentence_level(candidate, reference, rouge_n)
    # rouge is actually an f-score of the recall and precision 
    return(recall, precision, rouge)

#############
# Evaluation print-out example 
############# 
from IPython.display import Markdown, display

for i in range(10,15):
    printmd("**Generated summary:**"+decode_sequence(x_train[i].reshape(1,-1)))
    printmd("**Original summary:**"+seq2summary(y_train[i]))
    printmd("**Text:**"+seq2text(x_train[i]))
    printmd("**BLEU score(Unigram):** "+str(calc_indiv_BLEU(i, x_train, y_train)))
    rouge_n = 1 #this can be edited pending how we decide to evaluate ROUGE 
    printmd("**ROUGE-**"+str(rouge_n)+"**-Recall:** "+str(calc_indiv_rouge(i,x_train,y_train,rouge_n)[0]))
    printmd("**ROUGE-**"+str(rouge_n)+"**-Precision:** "+str(calc_indiv_rouge(i,x_train,y_train,rouge_n)[1]))
    printmd("**ROUGE-**"+str(rouge_n)+"**-Fscore:** "+str(calc_indiv_rouge(i,x_train,y_train,rouge_n)[2]))
    print('_________________________________________________________________')