import os
import json
from tqdm import tqdm
from database import Database

# cf
# season - 1:봄/가을, 2:여름, 3:겨울
# situation - 1:출근, 2:데이트, 3:행사(결혼식 등 공식모임), 4:사교모임(약속,파티), 5:일상생활, 6:레저 스포츠, 7:여행/휴가, 8:기타
# fit - 1:헐렁해보인다, 2:적당해보인다, 3:타이트해보인다
# gender - 1:남성, 2:여성
# age - 1:20-29, 2:30-39, 3:40-49, 4:50-60
# job - 1:전업주부, 2:기술/전문직, 3:판매/서비스직, 4:사무/관리직, 5:학생, 6:기타
# income - 삭제
# style1 - 1:화려하고 독특한 2:무난하고 평범한
# style2 - 1:남성적/여성적인 2:중성적인
# style3 - 1:전통적인 2:트랜디한
# style4 - 1:포멀한 2:캐주얼한
# style5 - 1:활발한 2:점잖은

SITUATION={1:'commute',2:'date',3:'official',4:'meeting',5:'daily',6:'sport',7:'trip',8:''}
SEASON={1:'spring/fall',2:'summer',3:'winter'}
FIT={1:'overfit',2:'nomalfit',3:'tightfit'}
JOB={1:'housewife',2:'technical/professional',3:'sales/service',4:'office/administrative',5:'student',6:''}


db=Database(host=None,user=None,password=None,db_name=None,port=None)

data_root='D:/recData/010.연도별 패션 선호도 파악 및 추천 데이터/01-1.정식개방데이터'

root_folders=os.listdir(data_root)

for root_folder in root_folders:
    print(f'{root_folder} start')
    target_folder=os.path.join(data_root,root_folder,'02.라벨링데이터')

    for target_file in tqdm(os.listdir(target_folder)):
        if target_file.split('.')[-1]=='json':
            with open(os.path.join(target_folder,target_file),'r') as j:
                data=json.load(j)

            # survey data
            columns=['uid','sid','rating','comment']
            uid=data['user']['R_id']
            sid=int(data['imgName'].split('_')[1])
            rating=data['item']['survey']['Q1']
            comment=SITUATION[data['item']['survey']['Q3']] if data['item']['survey'].get('Q3') else ''
            comment+=','+SEASON[data['item']['survey']['Q2']]
            comment+=','+FIT[data['item']['survey']['Q411']]
            comment=comment+',bright' if data['item']['survey']['Q412']==2 else comment
            comment=comment+',warm' if data['item']['survey']['Q413']==2 else comment
            comment=comment+',lightmood' if data['item']['survey']['Q414']==2 else comment
            comment=comment+',citysm' if data['item']['survey']['Q4202']==2 else comment
            comment=comment+',trendy' if data['item']['survey']['Q4203']==3 else comment
            comment=comment+',refined' if data['item']['survey']['Q4204']==4 else comment
            comment=comment+',neat' if data['item']['survey']['Q4205']==5 else comment
            comment=comment+',gorgeous' if data['item']['survey']['Q4206']==6 else comment
            comment=comment+',special' if data['item']['survey']['Q4207']==7 else comment
            comment=comment+',nomal' if data['item']['survey']['Q4208']==8 else comment
            comment=comment+',open_mind' if data['item']['survey']['Q4209']==9 else comment
            comment=comment+',practical' if data['item']['survey']['Q4210']==10 else comment
            comment=comment+',activity' if data['item']['survey']['Q4211']==11 else comment
            comment=comment+',cozy' if data['item']['survey']['Q4212']==12 else comment
            comment=comment+',lively' if data['item']['survey']['Q4213']==13 else comment
            comment=comment+',feminine' if data['item']['survey']['Q4214']==14 else comment
            comment=comment+',mascline' if data['item']['survey']['Q4215']==15 else comment
            comment=comment+',soft' if data['item']['survey']['Q4216']==16 else comment
            values=[uid,sid,rating,comment]
            db.insert_data(table_name='survey',columns=columns,values=values)

            # item data
            columns=['sid','item','style']
            sid=int(data['imgName'].split('_')[1])
            item=data['imgName']
            style=data['item']['style']
            values=[sid,item,style]
            db.insert_data(table_name='items',columns=columns,values=values,key=columns[0])

            # userinfo
            columns=['uid','style','sex','job','married']
            uid=data['user']['R_id']
            sex='man' if data['user']['r_gender']==1 else 'women'
            married='single' if data['user']['mar']==1 else 'married'
            job=JOB[data['user']['job']]
            style='colorful and unique' if data['user']['r_style1']==1 else 'decent and ordinary'
            style+=',masculine/feminine' if data['user']['r_style2']==1 else ',androgynous'
            style+=',traditional' if data['user']['r_style3']==1 else ',trendy'
            style+=',formal' if data['user']['r_style4']==1 else ',casual'
            style+=',active' if data['user']['r_style5']==1 else ',sedate'
            values=[uid,style,sex,job,married]
            db.insert_data(table_name='userinfo',columns=columns,values=values,key=columns[0])