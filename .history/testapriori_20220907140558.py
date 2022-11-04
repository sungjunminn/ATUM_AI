import pandas as pd
import numpy as np
train = pd.read_csv("C:\\Users\\become\\Desktop\\SortedData2\\Total\\df_total_rule.csv", header=0, index_col='INDEX')
print(train)
item_list = []
for i in range(len(train)):
    pd_train = train.loc[i].tolist()
    new_list = [x for x in pd_train if pd.isnull(x) == False]

    #print(new_list)
    item_list.append(new_list)
item_col_list = []
len(item_list)

for i in item_list:
    for j in i:
        item_col_list.append(j)

# all cases of biz item
unique_list = set(item_col_list)
unique_list = list(unique_list)
print(len(unique_list), unique_list)

import numpy as np

min_freq = 5
min_conf = 0.4

test_count = 0
total_list_1 = []
total_list_2 = []
total_list_3 = []
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
                test_list = []
                if len(list_a[i]) == 1:
                    test_list.append(list_a[i])
                    test_list.append(list_b[i])
                    test_list.append(list_freq[i])
                    test_list.append(list_conf[i])
                    total_list_1.append(test_list)
                elif len(list_a[i]) == 2:
                    test_list.append(list_a[i])
                    test_list.append(list_b[i])
                    test_list.append(list_freq[i])
                    test_list.append(list_conf[i])
                    total_list_2.append(test_list)
                elif len(list_a[i]) == 3:
                    test_list.append(list_a[i])
                    test_list.append(list_b[i])
                    test_list.append(list_freq[i])
                    test_list.append(list_conf[i])
                    total_list_3.append(test_list)


print(test_count)

col_name = ["list_a", "list_b", "list_freq", "list_conf"]
pd_total_list_1 = pd.DataFrame(total_list_1, columns=col_name)
pd_total_list_2 = pd.DataFrame(total_list_2, columns=col_name)
pd_total_list_3 = pd.DataFrame(total_list_3, columns=col_name)


print("입력 : ")
test_text = map(str, input().split())
list_test_text = list(test_text)
return_list = []
for i in range(len(pd_total_list_3)):

    if pd_total_list_3['list_a'][i] == list_test_text:
        print(pd_total_list_3['list_a'][i], pd_total_list_3['list_b'][i], pd_total_list_3['list_freq'][i], pd_total_list_3['list_conf'][i])
        list_three = [pd_total_list_3['list_b'][i], pd_total_list_3['list_freq'][i], pd_total_list_3['list_conf'][i]]
        return_list.append(list_three)

list_test_text = list_test_text[1:]

for i in range(len(pd_total_list_2)):

    if pd_total_list_2['list_a'][i] == list_test_text:
        print(pd_total_list_2['list_a'][i], pd_total_list_2['list_b'][i], pd_total_list_2['list_freq'][i], pd_total_list_2['list_conf'][i])
        list_two = [pd_total_list_2['list_b'][i], pd_total_list_2['list_freq'][i], pd_total_list_2['list_conf'][i]]
        return_list.append(list_two)

list_test_text = list_test_text[1:]

for i in range(len(pd_total_list_1)):

    if pd_total_list_1['list_a'][i] == list_test_text:
        print(pd_total_list_1['list_a'][i], pd_total_list_1['list_b'][i], pd_total_list_1['list_freq'][i],
              pd_total_list_1['list_conf'][i])
        list_one = [pd_total_list_1['list_b'][i], pd_total_list_1['list_freq'][i], pd_total_list_1['list_conf'][i]]
        return_list.append(list_one)

print("return_list : ", return_list)
