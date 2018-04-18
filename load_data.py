#!/usr/bin/env python3
# coding: utf-8
# File: load_data.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-17
import os, math
class DataLoader:
    def __init__(self):
        self.datafile = 'data/data.txt'
        self.dataset, self.cate_dict = self.load_data()
        self.catetype_dict = {
            '0': '汽车',
            '1': '财经',
            '2': 'IT',
            '3': '健康',
            '4': '体育',
            '5': '旅游',
            '6': '教育',
            '7': '招聘',
            '8': '文化',
            '9': '军事',
        }

    '''加载数据集'''
    def load_data(self):
        dataset = []
        cate_dict = {}
        for line in open(self.datafile):
            line = line.strip().split(',')
            cate = line[0]
            if cate not in cate_dict:
                cate_dict[cate] = 1
            else:
                cate_dict[cate] += 1
            dataset.append([line[0], [word for word in line[1].split(' ') if 'nbsp' not in word]])
        return dataset, cate_dict

# f = open('data.txt', 'w+')
# for root, dirs, files in os.walk('data/corpus'):
#     for file in files:
#         file_path = os.path.join(root, file)
#         for line in open(file_path):
#             line = line.strip()
#             if not line:
#                 continue
#             f.write(line + '\n')
# f.close()
