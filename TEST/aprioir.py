# -*- coding: utf-8 -*-
"""apriori.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mgaURlBGiiT0_0D1mJZd0bxZenxztXJh
"""

import os
import pandas as pd
item_list = []

os.chdir("C:\\Users\\become\\Desktop\\DB 데이터")
test_list = os.listdir()
'''
new_biz_item_list = pd.read_csv("C:\\Users\\become\\Desktop\\mldata\\old_recent_bizitem.csv", index_col=None, header=0)

old_biz = new_biz_item_list["old_item"].tolist()
new_biz = new_biz_item_list["recent_item"]
new_biz = new_biz.dropna()
new_biz = new_biz.tolist()
a = set(old_biz).intersection(set(new_biz))
print(a)
'''

for k in test_list:
    os.chdir("C:\\Users\\become\\Desktop\\DB 데이터\\"+k)

    tlist = os.listdir()

    file_list = []
    #print(tlist)
    for x in tlist:
        if ".csv" in x:
            file_list.append(x)

    for i in file_list:
        df = pd.read_csv(i)
        test_a = df["ITEMNAME"].values.tolist()
        item_list.append(test_a)
        '''
        if test_a in old_biz:
            pass
        else:
            item_list.append(test_a)
        '''
print(item_list)
item_col_list = []

for i in item_list:
    for j in i:
        item_col_list.append(j)

# all cases of biz item
unique_list = set(item_col_list)
unique_list = list(unique_list)
print(len(unique_list))

import numpy as np

min_freq = 15
min_conf = 0.8

test_count = 0
for carda in range(1, 4):
    for cardb in range(1, 4):
        # set data structure
        list_a = []
        list_b = []
        list_freq = []
        list_conf = []

        # for a rule (= biz item list)
        for candi_rule in item_list:
            # all cases of association rule
            for i in range(len(candi_rule) - carda - cardb):
                a = candi_rule[i:i + carda]
                b = candi_rule[i + carda:i + carda + cardb]

                if a in list_a and b in list_b:
                    # find exact index and increase number
                    for ii in range(len(list_a)):
                        if list_a[ii] == a and list_b[ii] == b:
                            list_freq[ii] = list_freq[ii] + 1
                            break

                else:
                    # append
                    list_a.append(a)
                    list_b.append(b)
                    list_freq.append(1)
                    list_conf.append(0)
        print(len(list_a), end=',')

        # remove association rules that are less than minimum frequency
        numbers = np.array(list_freq)
        del_tuple = np.where(numbers < min_freq)
        aa = del_tuple[0].tolist()
        for i in reversed(range(len(aa))):
            del list_a[aa[i]]
            del list_b[aa[i]]
            del list_freq[aa[i]]
            del list_conf[aa[i]]
        print(len(list_a), end=',')

        # calculate confidence
        # 1. count number of a
        num_a = [0 for i in range(len(list_a))]
        for candi_rule in item_list:
            # all cases of association rule
            for i in range(len(candi_rule) - carda - cardb):
                a = candi_rule[i:i + carda]
                if a in list_a:
                    tidx = list_a.index(a)
                    num_a[tidx] = num_a[tidx] + 1

        # A can be redundant
        for i in range(len(list_a)):
            for j in range(i + 1, len(list_a)):
                if list_a[i] == list_a[j] and num_a[j] == 0:
                    num_a[j] = num_a[i]

        # 2. fill in confidence variable
        for i in range(len(list_a)):
            list_conf[i] = list_freq[i] / num_a[i]

        # remove association rules that are less than minimum confidence
        numbers = np.array(list_conf)
        del_tuple = np.where(numbers < min_conf)
        aa = del_tuple[0].tolist()
        for i in reversed(range(len(aa))):
            del list_a[aa[i]]
            del list_b[aa[i]]
            del list_freq[aa[i]]
            del list_conf[aa[i]]
        print(len(list_a))

        print('size of a: {}, size of b: {}, num of rules: {}'.format(carda, cardb, len(list_a)))
        if len(list_a) != 0:
            test_count += len(list_a)
            for i in range(len(list_a)):
                print('{},{},{},{:.3f}'.format(list_a[i], list_b[i], list_freq[i], list_conf[i]))

print("전체 범위 내 조건을 만족한 rule의 수 : ", test_count)
