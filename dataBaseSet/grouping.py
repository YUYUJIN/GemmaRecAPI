import os
import pandas as pd
from dotenv import load_dotenv
import tqdm

from config import *

load_dotenv()
args.host,args.user,args.password,args.dbname,args.port=os.environ.get('HOST'),os.environ.get('USERNAME'),os.environ.get('PASSWARD'),os.environ.get('DATABASENAME'),int(os.environ.get('PORT'))
database=set_template(args)

# category
sex=['man','women']
married=['married','single']
job=['housewife','technical/professional','sales/service','office/administrative','student','']
style1=['decent and ordinary','colorful and unique']
style2=['masculine/feminine','androgynous']
style3=['traditional','trendy']
style4=['formal','casual']
style5=['active','sedate']
style=[]
for style1_ in style1:
    for style2_ in style2:
        for style3_ in style3:
            for style4_ in style4:
                for style5_ in style5:
                    style.append(f'{style1_},{style2_},{style3_},{style4_},{style5_}')

group_inv={}
index=1
for sex_ in sex:
    for married_ in married:
        for job_ in job:
            for style_ in style:
                rows=database.get_data('userinfo','uid',option=f'sex=\'{sex_}\' and married=\'{married_}\' and job=\'{job_}\' and style=\'{style_}\'')
                if len(rows)==0:
                    pass
                else:
                    database.insert_data('usergroup',['uid','style','sex','job','married'],[index,style_,sex_,job_,married_],'uid')
                    for row in rows:
                        group_inv[row[0]]=index
                    index+=1

rows=database.get_data('survey')
for row in tqdm.tqdm(rows):
    database.insert_data('surveygroup',['uid','sid','rating','comment'],[group_inv[row[0]],row[1],row[2],row[3]])
