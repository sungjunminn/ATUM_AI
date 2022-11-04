import pandas as pd

train = pd.read_csv("C:\\Users\\become\\Desktop\\Final_Rule_Sequence.csv", header=0, index_col='INDEX')
print(train)
total_name_list = []
for i in range(len(train)):
    pd_train = train.loc[i].tolist()
    new_list = [x for x in pd_train if pd.isnull(x) == False]

    # print(new_list)
    item_name_list = new_list
    print("전", item_name_list)
    for z in range(len(item_name_list)):
        if item_name_list[z] == "IFBLOCK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "ELSEBLOCK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "ELSEIFBLOCK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "WHILELOOP-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "FORLOOP-BODY":
            item_name_list[z] = "0"
    while "0" in item_name_list:
        item_name_list.remove("0")
    print("후", item_name_list)
    total_name_list.append(item_name_list)
print(total_name_list)
pd_total_list = pd.DataFrame(total_name_list)
print(pd_total_list)
#pd_total_list.to_csv("C:\\Users\\become\\Desktop\\test_name4.csv")