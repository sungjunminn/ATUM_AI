from TEST import mariadb

"""
mariadb 연결 테스트

"""

config = {
  'host': '192.168.0.28',
  'port': 3306,
  'user': 'ATUM_DEV_ENHANCE',
  'password': 'become1!',
  'database': 'ATUM_DEV_ENHANCE'


}

conn = mariadb.connect(**config)

cur = conn.cursor()
print()
cur.execute("select * from CHATBOT")

row_headers = [x[0] for x in cur.description]
rv = cur.fetchall()
print(rv)
json_data=[]
for result in rv:
  print(result)

# str = json.dumps(json_data)
# print(str)
