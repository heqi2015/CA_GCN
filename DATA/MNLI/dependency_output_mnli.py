## Input: MNLI Dataset file 
#         traindata.tsv, devdata_matched.tsv, devdata_mismatched.tsv
# Such as,
# and the professors who go there and you're not going to see the professors you know you're going to see some TA you know uh	You don't really see the TAs.	2
#
## Output: Dependency results of input file
#          dependency_mnli_train.tsv/dependency_mnli_dev_matched.tsv/dependency_mnli_dev_mismatched.tsv
# Such as,
# cc	going	11	and	1
# det	professors	3	the	2
# root	_ROOT	0	professors	3
# nsubj	go	5	who	4
# acl:relcl	professors	3	go	5
# advmod	go	5	there	6
# cc	going	11	and	7
# nsubj	going	11	you	8
# aux	going	11	're	9
# advmod	going	11	not	10
# conj	go	5	going	11
# mark	see	13	to	12
# xcomp	going	11	see	13
# det	professors	15	the	14
# obj	see	13	professors	15
# nsubj	know	17	you	16
# acl:relcl	professors	15	know	17
# nsubj	going	20	you	18
# aux	going	20	're	19
# ccomp	know	17	going	20
# mark	see	22	to	21
# xcomp	going	20	see	22
# det	TA	24	some	23
# obj	see	22	TA	24
# nsubj	know	26	you	25
# acl:relcl	TA	24	know	26
# discourse	know	17	uh	27

import argparse
import glob
import logging
import os
import random
import csv

import numpy as np
import stanza

DATA_QQP = []
labels = []
with open('devdata_matched.tsv') as fr:
    for line in fr.readlines():
        line = line.strip('\n').split('\t')
        DATA_QQP.append(line)

# DATA_QQP.remove(DATA_QQP[0])

stanza.download('en')
nlp = stanza.Pipeline()

fo = open("dependency_mnli_test_matched.tsv", "w")

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

for i, data in enumerate(DATA_QQP):
    # if i % 1000 == 0:
    #     print("process: " + str(i) + " data done!")
    get_dependencies(data[0])
    fo.write("\n")
    get_dependencies(data[1])
    fo.write("\n")

fo.close()

# labels = np.array(labels)
# np.savetxt('train_labels.txt', labels, fmt='%d', delimiter=',')