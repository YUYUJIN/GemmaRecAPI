from pathlib import Path
import pickle
import shutil
import tempfile
import os
import json

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class SurveyDataset:
    def __init__(self, args, database):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.database=database

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    def load_dataset(self):
        print('----loading data from database----')
        rows = self.database.get_data(table_name='surveygroup')
        df=pd.DataFrame(data=rows,columns=['uid','sid','rating','comment'])
        item,info,comment = self.load_meta_dict()
        df = df[df['sid'].isin(item)]  # filter items without meta info
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        item = {smap[k]: v for k, v in item.items() if k in smap}
        info = {umap[k]: v for k, v in info.items() if k in umap}
        comment={umap[k]: v for k, v in comment.items() if k in umap}
        for k in comment.keys():
            comment[k]={smap[q]: v for q, v in comment[k].items() if q in smap}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'info': info,
                   'item': item,
                   'comment': comment,
                   'umap': umap,
                   'smap': smap}
        return dataset

    def load_meta_dict(self):
        # items
        rows = self.database.get_data(table_name='items')
        df=pd.DataFrame(data=rows,columns=['sid','item','style'])
        item_dict = {}
        for row in df.itertuples():
            title = row[2][:-4]  # remove format (optional)
            imageFomat = row[2][-4:] # maybe use

            # title = re.sub('\(.*?\)', '', title).strip()
            # # the rest articles and parentheses are not considered here
            # if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
            #     title_pre = title.split(', ')[:-1]
            #     title_post = title.split(', ')[-1]
            #     title_pre = ', '.join(title_pre)
            #     title = title_post + ' ' + title_pre

            item_dict[row[1]] = title
        
        # info
        rows = self.database.get_data(table_name='usergroup')
        df=pd.DataFrame(data=rows,columns=['uid','style','sex','job','married'])
        info_dict={}
        comment_dict={}
        for row in df.itertuples():
            info=f'{row[3]}, {row[5]}, {row[4]}, {row[2]}'
            info_dict[row[1]]=info
            comment_dict[row[1]]={}

        # comment
        rows = self.database.get_data(table_name='surveygroup')
        df=pd.DataFrame(data=rows,columns=['uid','sid','rating','comment'])        
        for row in df.itertuples():
            comment=row[4]
            comment_dict[row[1]][row[2]]=comment

        
        return item_dict,info_dict,comment_dict
    
    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 1 or self.min_uc > 1:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
                if self.min_sc > 1:
                    item_sizes = df.groupby('sid').size()
                    good_items = item_sizes.index[item_sizes >= self.min_sc]
                    df = df[df['sid'].isin(good_items)]

                if self.min_uc > 1:
                    user_sizes = df.groupby('uid').size()
                    good_users = user_sizes.index[user_sizes >= self.min_uc]
                    df = df[df['uid'].isin(good_users)]

                item_sizes = df.groupby('sid').size()
                good_items = item_sizes.index[item_sizes >= self.min_sc]
                user_sizes = df.groupby('uid').size()
                good_users = user_sizes.index[user_sizes >= self.min_uc]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        print('Splitting')
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(
            lambda d: list(d['sid']))
        train, val, test = {}, {}, {}
        for i in range(user_count):
            user = i + 1
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        return train, val, test
