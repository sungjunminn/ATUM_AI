import os
import pandas as pd

train = pd.read_csv("C:\\Users\\become\\Desktop\\new_db_data\\ATUM_DEV_MESSTD_ATUM\\"
                    "BR_TxnLogTraceSave.csv", index_col=0,
                    header=0)

print(train)

index_list = train["d"].tolist()
name_list = []
name = train["ITEMNAME"].tolist()

for i in range(len(train)):
    name_list.append(name[index_list[i]])

print(name_list)
name_list = [name_list]
pd_name_list = pd.DataFrame(name_list)

pd_name_list.to_csv("C:\\Users\\become\\Desktop\\DB 데이터\\pd.csv")