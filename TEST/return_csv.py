
import os
import pandas as pd
item_list = []

os.chdir("C:\\Users\\become\\Desktop\\new_db_data\\")
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

total_name_list = []
name_list = []
for k in test_list:
    os.chdir("C:\\Users\\become\\Desktop\\new_db_data\\"+k)

    tlist = os.listdir()

    file_list = []
    #print(tlist)
    for x in tlist:
        if ".csv" in x:
            file_list.append(x)

    for i in file_list:
        train = pd.read_csv(i)
        visited = [False] * len(train)
        total_list = []

        SUPER_ID = train["SUPERITEMID"].tolist()

        def dfs_test(ITEM_ID):
            global SUPER_ID

            index = []
            for k in range(len(SUPER_ID)):

                if ITEM_ID == SUPER_ID[k]:
                    index.append(k)

            index = sorted(index, reverse=False)

            return index


        def dfs_test_2(index):
            global visited, SUPER_ID

            if index not in total_list:
                total_list.append(index)

            if train["ITEMID"][index] in SUPER_ID:
                index_list = dfs_test(train["ITEMID"][index])
                #print("index_list", index_list)
                # index_list.append(SUPER_ID.index(train["ITEMID"][index]))
                visited[SUPER_ID.index(train["ITEMID"][index])] = True
                return dfs_test_2(SUPER_ID.index(train["ITEMID"][index]))

        while 1:
            for y in range(len(train)):
                #print(total_list)
                if y not in total_list:
                    total_list.append(y)
                    new_index = dfs_test(train["ITEMID"][y])

                    for j in range(len(new_index)):
                        dfs_test_2(new_index[j])
                else:
                    continue
            #print("total : ", total_list)
            if len(total_list) >= len(train):
                break
        item_name_list = []
        test1 = train['ITEMNAME'].tolist()
        for a in range(len(total_list)):
            item_name_list.append(test1[total_list[a]])

        print("전", item_name_list)
        for z in range(len(item_name_list)):
            if item_name_list[z] == "EFORLOOP":
                item_name_list[z] = "FORLOOP"
            if item_name_list[z] == "EFORLOOP-BODY":
                item_name_list[z] = "FORLOOP-BODY"
            if item_name_list[z] == "EFOR-MAINBLOCK":
                item_name_list[z] = "FOR-MAINBLOCK"
            if item_name_list[z] == "SETSUCESSCOUNT":
                item_name_list[z] = "SETSUCCESSCOUNT"
            if item_name_list[z] == "CONDITIONALLOGERROR":
                item_name_list[z] = "VALIDATIONCHECK"
            if item_name_list[z] == "VALIDATIONCHECK-BODY":
                item_name_list[z] = "0"
            if item_name_list[z] == "MAPPUT-BODY":
                item_name_list[z] = "0"
            if item_name_list[z] == "if":
                item_name_list[z] = "0"
            if item_name_list[z] == "ConditionalUDF-BODY":
                item_name_list[z] = "0"
            if item_name_list[z] == "DBOBJECTSETS-BODY":
                item_name_list[z] = "0"
        while "0" in item_name_list:
            item_name_list.remove("0")
        print("후", item_name_list)
        total_name_list.append(item_name_list)
        name = k + "/" +i
        name_list.append(name)
#print(total_name_list)

pd_total_list = pd.DataFrame(total_name_list)
#print(pd_total_list)
#print(name_list)
pd_name_list = pd.DataFrame(name_list)
#pd_name_list.to_csv("C:\\Users\\become\\PycharmProjects\\pythonProject\\name_list.csv")
#pd_total_list.to_csv("C:\\Users\\become\\PycharmProjects\\pythonProject\\test_name3.csv")
