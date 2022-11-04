import pymysql

config = {
    'host': '192.168.0.28',
    'port': 3306,
    'user': 'ATUM_DEV_ENHANCE',
    'password': 'become1!',
    'database': 'ATUM_DEV_ENHANCE'
}

# conn = pymysql.connect(**config)
conn = pymysql.connect(host='192.168.0.28', port=3306,
                       user='ATUM_DEV_ENHANCE', password='become1!', database='ATUM_DEV_ENHANCE')
cur = conn.cursor()

cur.execute("select * from CHATBOT")
result = cur.fetchall()
print(result)