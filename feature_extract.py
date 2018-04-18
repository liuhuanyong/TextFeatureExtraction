#!/usr/bin/env python3
# coding: utf-8
# File: feature_extract.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-18

import math
from load_data import *

class FeatureExtract:
    def __init__(self):
        self.dataset, self.cate_dict = DataLoader().load_data()
        self.cate_nums = len(self.cate_dict)
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

    '''统计文本的类别df特征'''
    def collect_dfdict(self):
        worddf_dict = {}
        for data in self.dataset:
            category = data[0]
            for word in set(data[1]):
                if word not in worddf_dict:
                    worddf_dict[word] = category
                else:
                    worddf_dict[word] += '@' + category

        for word, word_category in worddf_dict.items():
            cate_dict = {}
            for cate in word_category.split('@'):
                if cate not in cate_dict:
                    cate_dict[cate] = 1
                else:
                    cate_dict[cate] += 1
            worddf_dict[word] = cate_dict
        return worddf_dict

    '''统计文本出现的文档数量，作为特征选择的一种方法，得到的是一个与类别无关的全局特征'''
    def DF(self, feature_num):
        df_dict = {}
        for data in self.dataset:
            for word in set(data[1]):
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
        df_dict = sorted(df_dict.items(), key=lambda asd:asd[1], reverse=True)[:feature_num]
        features = [item[0] for item in df_dict]
        return features

    '''CHI，作为特征选择的一种方法，计算对应的A,B,C,D，卡方值的计算公式：(N*(A*D - B*C)**2)/((A+C)*(A+B)*(B+D)*(B+C))，得到的是一个类别的局部特征'''
    def CHI(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        chi_dict = {}
        for word, word_cate in worddf_dict.items():
            data = {}
            for cate in range(self.cate_nums):
                cate = str(cate)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                chi_score = (N*(A*D - B*C)**2)/((A+C)*(A+B)*(B+D)*(B+C))
                data[cate] = chi_score
            chi_dict[word] = data

        features = self.select_best(feature_num, chi_dict)
        return features

    '''IG，信息增益，计算对应的A,B,C,D,得到的是一个全局的特征'''
    def IG(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        ig_dict = {}
        for word, word_cate in worddf_dict.items():
            HC = 0.0  #原先分类系统的信息熵
            HTC = 0.0 #分类系统包含该词的信息熵
            HT_C = 0.0 #分类系统不包含该词的信息熵
            for cate in range(self.cate_nums):
                cate = str(cate)
                N1 = self.cate_dict[cate]
                hc = -(N1/N) * math.log(N1/N)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                # 在这里会出现缺失值的情况，如果不做平滑会受语料大小影响效果，这里作简单add-one平滑处理
                p_t = (A + B) / N
                p_not_t = (C + D)/ N
                p_t_c = (A + 1)/ (A + B + self.cate_nums)
                p_t_not_c = (C + 1)/ (C + D + self.cate_nums)
                h_t_ci = p_t * p_t_c * math.log(p_t_c)
                h_t_not_ci = p_not_t * p_t_not_c * math.log(p_t_not_c)
                #针对所有类别，对信息熵作累加操作
                HC += hc
                HTC += h_t_ci
                HT_C += h_t_not_ci
            ig_score = HC + HTC + HT_C
            ig_dict[word] = ig_score

        ig_dict = sorted(ig_dict.items(), key=lambda asd:asd[1], reverse=True)[:feature_num]
        features = [item[0] for item in ig_dict]
        return features

    '''MI, 互信息，计算词语与类别之间的相关性，得到的是一个局部特征'''
    def MI(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        mi_dict = {}
        for word, word_cate in worddf_dict.items():
            data = {}
            for cate in range(self.cate_nums):
                cate = str(cate)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                #要进行平滑处理, 计算条件概率时，进行简单add-one平滑处理
                p_t_c = ( A + 1)/(A + B + self.cate_nums)
                p_c = (A + C) / N
                p_t = (A + B) / N
                mi_score = p_t_c * math.log(p_t_c / (p_c * p_t))
                data[cate] = mi_score
            mi_dict[word] = data

        features = self.select_best(feature_num, mi_dict)
        return features

    '''基于得到的词-类别相关性字典，统计每一类对应的top词，组合成为特征词'''
    def select_best(self, feature_num, word_dict):
        cate_worddict = {}
        features = []
        for word, scores in word_dict.items():
            for cate, word_score in scores.items():
                if cate not in cate_worddict:
                    cate_worddict[cate] = {}
                else:
                    cate_worddict[cate][word] = word_score
        #为了防止类别之间的特征词有重复，要加大分配的topnum
        top_num = int(feature_num/self.cate_nums) + 100

        for cate, words in cate_worddict.items():
            words = sorted(words.items(), key=lambda asd:asd[1], reverse=True)[:top_num]
            top_words = [item[0] for item in words]
            features += top_words

        return list(set(features[:feature_num]))

# dataer = FeatureExtract()
# features = dataer.DF(5000)
#
# f = open('data/df.txt', 'w+')
# f.write('\n'.join(features))
# f.close()