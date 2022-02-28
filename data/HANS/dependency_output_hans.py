## Input: heuristics_evaluation_set.txt -- HANS Dataset downloaded from Official Website
#  Such as 
# non-entailment	( ( The president ) ( ( advised ( the doctor ) ) . ) )	( ( The doctor ) ( ( advised ( the president ) ) . ) )	(ROOT (S (NP (DT The) (NN president)) (VP (VBD advised) (NP (DT the) (NN doctor))) (. .)))	(ROOT (S (NP (DT The) (NN doctor)) (VP (VBD advised) (NP (DT the) (NN president))) (. .)))	The president advised the doctor .	The doctor advised the president .	ex0	lexical_overlap	ln_subject/object_swap	temp1
#
## Output: dedependency_hans.tsv -- Dependency results of input file
# Such as
# det	president	2	The	1
# nsubj	advised	3	president	2
# root	_ROOT	0	advised	3
# det	doctor	5	the	4
# obj	advised	3	doctor	5
# punct	advised	3	.	6

import argparse
import glob
import logging
import os
import random
import csv

import numpy as np
import stanfordnlp

hans = []
labels = []
with open('heuristics_evaluation_set.txt') as fr:
    for line in fr.readlines():
        line = line.strip('\n').split('\t')
        hans.append(line)

hans.remove(hans[0])

stanfordnlp.download('en', force=True)
nlp = stanfordnlp.Pipeline()

fo = open("dependency_hans.tsv", "w")

def get_dependencies(text):
    if text == None or len(text) < 1:
        fo.write("\n")
        return
    length = 0
    doc = nlp(text)
    for i in range(len(doc.sentences)):
        for idx, dep_edge in enumerate(doc.sentences[i].dependencies):
            x = "_ROOT"
            y = 0
            if dep_edge[0].index != '0':
                x = doc.sentences[i].dependencies[int(dep_edge[0].index) - 1][2].text
                y = int(dep_edge[0].index) + length
            fo.write(dep_edge[1] + "\t" + x + "\t" + str(y) + "\t" + dep_edge[2].text + "\t" + str(idx + length + 1) + "\n")
        length = length + len(doc.sentences[i].dependencies)

for i, data in enumerate(hans):
    # if i % 1000 == 0:
    #     print("process: " + str(i) + " data done!")
    # labels.append(float(data[9]))
    get_dependencies(data[5])
    fo.write("\n")
    get_dependencies(data[6])
    fo.write("\n")

fo.close()

# labels = np.array(labels)
# np.savetxt('test_labels.txt', labels, delimiter=',')