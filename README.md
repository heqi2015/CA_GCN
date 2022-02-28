# CA_GCN
The code for paper "Enhancing generalization in natural language inference by syntax." Findings of the Association for Computational Linguistics: EMNLP 2020. 

Sorry for late updates, as I started a new career and changed my research direction in the last year, and too many trivial things take up my time. Thanks to my colleague Qingjing for helping organize and test the code.

Please train the model by running train_mnli.py, and the result on MNLI will be obtained. The result on HANS can be achieved by referring to evaluate_heur_output.py. 

It is worth mentioning that a simply training strategy is adopted in our codes, and the experimental results can be even further improved by adjusting the lr decay strategy of both the BERT and GCN parts, and improving the training epoch number.  
