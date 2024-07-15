import psycopg2

class Database:
    def __init__(self,host,port,user,password,db_name):
        print(f'{host}\'s {db_name} is connected')
        # Postgre Connection
        self.connect=psycopg2.connect(host=host,dbname=db_name,user=user,password=password,port=port)

        self.cursor=self.connect.cursor() # cursor

    def get_data(self,table_name,targets='*',option=None):
        curs=self.cursor

        if option==None:
            sql=f'select {targets} from {table_name};'
        else:
            sql=f'select {targets} from {table_name} where {option};'

        curs.execute(sql)
        rows=curs.fetchall()
        
        return rows
    
    def insert_data(self,table_name,columns,values,key=None):
        curs=self.cursor

        column=''
        for c in columns:
            column+=f'{c},'
        column=column[:-1]

        options=''
        for v in values:
            options+='%s,'
        options=options[:-1]
        
        if key==None:
            sql=f'insert into {table_name} ({column}) values ({options});'
        else:
            sql=f'insert into {table_name} ({column}) values ({options}) on conflict({key}) do nothing;'
        curs.execute(sql,values)
        self.connect.commit()

    def update_data(self,table_name,columns,values):
        curs=self.cursor

        column=''
        for c in columns:
            column+=f'{c},'
        column=column[:-1]

        options=''
        for v in values:
            options+='$s,'
        options=options[:-1]

        sql=f'update into {table_name} ({column}) values ({options});'
        curs.execute(sql,values)
        self.connect.commit()

    def delete_data(self,table_name,options):
        curs=self.cursor

        sql=f'delete form {table_name} where {options};'
        curs.execute(sql)
        self.connect.commit()

    def db_disconnect(self):
        self.cursor.close()
        self.connect.close()

    def __del__(self):
        self.db_disconnect()
