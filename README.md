# CA_GCN
The code for paper "Enhancing generalization in natural language inference by syntax." Findings of the Association for Computational Linguistics: EMNLP 2020. 

Sorry for late updates, as I started a new career and changed my research direction in the last year, and too many routine but trivial things take up my time. Thanks to my colleague Qingjing Fei for helping organize and test the code.

Please train the model by running train_mnli.py, and the result on MNLI will be obtained. The result on HANS can be achieved by referring to evaluate_heur_output.py. 

It is worth mentioning that a simply training strategy is adopted in our codes, and the experimental results can be even further improved by adjusting the lr decay strategy of both the BERT and GCN parts, or increasing the training epoch number.  


Steps:
1. Download the HANS dataset (heuristics_evaluation_set.txt), and save it in the folder "data/HANS/". 
2. Download the MNLI dataset, and save it in the folder "data/MNLI/". 
3. Run the file data/HANS/dependency_output_hans.py and data/MNLI/dependency_output_mnli.py, and generate the .tsv file which hold the dependency relationship. 
4. Run the train_mnli.py to train the model and the evaluation results on MNLI. Particularly, for the first time running of train_mnli.py, please set the parameter force_re_parsing_depen=True for the function fetch_data_and_graph, which forces generating the .pt file from the .tsv file and hold the dependency relationship in a tensor form. Note that the seed setting has an impact on the performance.

Test on HANS:
1. Set hyps["eval_on_hans"]=True in train_mnli.py. 
2. Set hyps["eval_model_folder"] to a folder containing a trained model, such as "results/bert_co_attn/seed69/lr1e-04_202202171618". Set hyps["eval_model_path"] to the saved model such as obtained in the last epoch, and hyps["eval_data_list"] to the .tsv file such as "data/HANS/dependency_hans.tsv".
3. Run train_mnli.py, and obtain the result on HANS, which is saved in a .txt form under the folder hyps["eval_model_folder"]. 
4. Calculate the index on HANS by "python evaluate_heur_output.py [hans_preds_file] > [result_file]" where [hans_preds_file] is set to hyps["eval_hans_result"] with the absolute path. 
