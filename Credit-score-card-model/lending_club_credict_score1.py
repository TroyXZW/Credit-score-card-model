#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

basic_features = np.array(['loan_status', 'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc', \
                           'verification_status', 'purpose', 'addr_state', 'dti', 'delinq_2yrs',
                           'mths_since_last_delinq', \
                           'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'open_il_24m', 'inq_last_12m',
                           'mths_since_recent_bc_dlq', \
                           'num_accts_ever_120_pd', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'grade'])

# 一.数据集生成
print('--------------------数据生成--------------------\n')
dataset = pd.read_csv('E:\PycharmProjects\sklearn\Data\风控\LoanStats_2017Q1.csv', encoding='utf-8', skiprows=1, usecols=basic_features)
for i in range(1, 5):  # 将2017年的数据集进行合并
    data = pd.read_csv('LoanStats_2017Q{}.csv'.format(i), encoding='utf-8', skiprows=1, usecols=basic_features)
    dataset = pd.concat([dataset, data.iloc[1:, :]], axis=0)  # .iloc[1:, :]]跳过表头
dataset = dataset.loc[(dataset['loan_status'] == 'Charged Off') | (dataset['loan_status'] == 'Fully Paid')]  # 只选出带标签的样本
dataset = dataset.reset_index(drop=True)  # 将索引重置
loan_status = dataset['loan_status']  # 将目标变量放在第一列
del dataset['loan_status']
dataset.insert(0, 'target', loan_status)
dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\风控\loan_data.csv', index=False)
print('数据集生成成功\n')

print("DONE")