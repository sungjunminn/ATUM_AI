import pandas as pd

train = pd.read_csv("../Module/test_name4.csv", index_col=0, header=0)
print(train)
list_rule_block = []
for i in range(len(train)):
    pd_train = train.loc[i].tolist()
    new_list = [x for x in pd_train if pd.isnull(x) == False]

    # print(new_list)
    list_rule_block.append(new_list)
print(list_rule_block)
item_col_list = []
# len(item_list)

for i in list_rule_block:
    for j in i:
        item_col_list.append(j)
print(item_col_list)
# all cases of biz item
unique_list = set(item_col_list)

list_block = list(unique_list)

print(unique_list, len(unique_list))
list_rule = len(train)
print(list_rule)
dic_block_num = {}
for idx in range(len(list_block)):
    dic_block_num[list_block[idx]] = idx
print(dic_block_num)