#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
此脚本用于定义函数，供1，2，3调用
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def f(x):
    """时间转化"""
    if 'years' in str(x):
        x = str(x).strip('years')
        x = str(x).replace('+', '')
    elif '<' in str(x):
        x = str(x).strip('year')
        x = str(x).strip('<')
        x = float(x) - 1
    else:
        x = str(x).strip('year')
    return float(x)


def drop_duplicate(x, i):
    """去除每一个特征中重复值高于百分之95的特征"""
    if ((x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum()) > 0.95):
        print('去除的特征为{0}，该特征内重复值占比为{1:.4f}\n'.format(i, (
                x.value_counts().sort_values(ascending=False).max() / x.value_counts().sum())))
        del x


def Regroup(df, col):
    """准备待分箱的数据"""
    total = df.groupby([col])['target'].count()
    df_total = pd.DataFrame({'total': total})
    bad = df.groupby([col])['target'].sum()
    df_bad = pd.DataFrame({'bad': bad})
    regroup = df_total.merge(df_bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    return regroup


def Chi2(df, total, bad, overallRate):
    """#此函数计算卡方值
     :df dataFrame
     :total_col 每个值得总数量
     :bad_col 每个值的坏数据数量
     :overallRate 坏数据的占比
     : return 卡方值"""
    df2 = df.copy()
    df2['good_col'] = df2[total] - df2[bad]
    df2['expected_bad'] = df[total].apply(lambda x: x * overallRate)
    df2['expected_good'] = df[total].apply(lambda x: x * (1 - overallRate))
    combined_bad = zip(df2['expected_bad'], df2[bad])
    combined_good = zip(df2['expected_good'], df2['good_col'])
    chi1 = [(i[0] - i[1]) ** 2 / i[0] for i in combined_bad]
    chi1 = sum(chi1)
    chi2 = [(i[0] - i[1]) ** 2 / i[0] for i in combined_good]
    chi2 = sum(chi2)
    chi3 = chi1 + chi2
    return chi3


def ChiMerge_MaxInterval_Original(df, col, overallRate, max_interval=5):
    """分箱原则：最大分箱数分箱"""
    total = df.groupby([col])['target'].count()
    df_total = pd.DataFrame({'total': total})
    bad = df.groupby([col])['target'].sum()
    df_bad = pd.DataFrame({'bad': bad})
    regroup = df_total.merge(df_bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    # return regroup
    colLevels = regroup[col]
    colLevels = sorted(list(colLevels))
    # 转化为列表的形式
    groupIntervals = [[i] for i in colLevels]
    groupNum = len(groupIntervals)
    while (len(groupIntervals) > max_interval):
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)
        min_position = chisqList.index(min(chisqList))
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
            # 合并箱体
        groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]  # 列表相加
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    groupIntervals = [sorted(i) for i in groupIntervals]
    # print (groupIntervals)
    # cutOffPoints=[i[-1] for i in groupIntervals[:-1]]
    return groupIntervals, regroup  # 返回元组


def BadRateMonotone(df, col, L):
    """分箱以后检查每箱的bad_rate的单调性，如果不满足，那么继续进行相邻的两项合并，直到bad_rate单调为止"""
    R = []
    for interval in L:
        df2 = df.loc[df[col].isin(interval)]
        rate = sum(df2['bad']) / sum(df2['total'])
        R.append(rate)
    badRateMonotoneA = [R[i] < R[i + 1] for i in range(len(R) - 1)]
    badRateMonotoneD = [R[i] > R[i + 1] for i in range(len(R) - 1)]
    MonotoneA = len(set(badRateMonotoneA))
    MonotoneD = len(set(badRateMonotoneD))
    if MonotoneA == 1 or MonotoneD == 1:
        return True
    else:
        return False
