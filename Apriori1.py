import pandas as pd
import numpy as np

address = pd.read_csv(r'C:/Users/become/Desktop/Apriori_result/mldata/itemaddress')
address_lst = address['ADDRESS1'].tolist()

dataset = []

for a in range(len(address_lst)):
  mldata = pd.read_csv(address_lst[a])
  data = pd.DataFrame(mldata)

  level0 = data[data['LEVEL'] == 0]
  levelnot0 = data[data['LEVEL'] != 0]
  df_level0 = level0.sort_values(by='SEQNUM', ascending = True)
  df = pd.concat([df_level0, levelnot0], ignore_index=True)
  df.reset_index(inplace = True)

  column = ['GROUP_SEQ','RULEID','ITEMID','LEVEL','SUPERITEMID','SEQNUM','ITEMNAME']
  items = pd.DataFrame(columns=column)

  for i in range(len(df)):
    for j in range(len(df)):
      arr = pd.DataFrame(columns=column)
      if df['LEVEL'][j] == i:
        aa = df.iloc[j:j+1]
        arr = arr.append(aa, ignore_index = True)
        if i == 0:
          items = items.append(arr, ignore_index = True)
        else:
          num = items[items['ITEMID'] == df['SUPERITEMID'][j]].index[0]
          items = pd.concat([items.iloc[:num+1], df.iloc[j:j+1], items.iloc[num+1:]], ignore_index = True)

  words = address['NO'].tolist()
  final_df1 = items[~items['ITEMID'].map(lambda x: all(word in x for  word in words))]
  final_df = final_df1[~final_df1['ITEMNAME'].str.contains('DBOBJECTSETS-BODY|ELSEBLOCK-BODY|VALIDATIONCHECK-BODY|FORLOOP-BODY')]
  final_lst = final_df['ITEMNAME'].tolist()
  dataset.append(final_lst)


#Apriori, Association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset_df = pd.DataFrame(dataset)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support = 0.7, use_colnames=True, max_len=3)

#antecedents : 조건, consequents : 결과, support : 지지도(전체 항목중 x,y를 포함하는 경우의 비율), confidence : 신뢰도(x가 있을 때 y도 있는 비율 - 조건부 확률), lift: 향상도(우연적 기회를 벗어나기 위한 값) 신뢰도가 같을 때 두 개중 어떤 것이 다음에 나올지
af = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
af