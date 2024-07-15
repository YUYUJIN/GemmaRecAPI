import psycopg2

db=psycopg2.connect(host='localhost',dbname='llamarec',user='user',password='1234',port=5432)

cursor=db.cursor()

s='test'
cursor.execute(f'insert into survey values (1,1,1,{s})')
db.commit()

cursor.execute('select * from survey')
result=cursor.fetchall()
print(result)

cursor.close()
db.close()