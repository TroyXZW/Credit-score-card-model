# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import Fuction_Total as FT

# 消除版本引起的参数设置警告
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 为了直观的显示数字，不采用科学计数法

# 设置图标可以显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示字符
sns.set(font="simhei")

# 二.数据集导入
print('--------------------数据导入--------------------\n')
loan_data = pd.read_csv('E:\PycharmProjects\sklearn\Data\风控\loan_data.csv', encoding='utf-8')
'''loan_data.rename(columns = {'target':'目标','loan_amnt':'贷款金额','term':'还款期限','emp_length':'工作年限','home_ownership':'房屋拥有情况',\
						  'annual_inc':'年收入','verification_status':'是否被核实','purpose':'贷款目的','addr_state':'所在州','dti':'每月需要偿还的债务/月收入',\
						  'delinq_2yrs':'过去2年中逾期30天以上违约事件的次数','mths_since_last_delinq':'自上次违约以来的月数','open_acc':'借款人未结信用产品的数目',\
						  'pub_rec':'公共不良记录的数量','total_acc':'借款人当前的信用产品总数','acc_now_delinq':'借款人现在拖欠的帐户数',\
						  'open_il_24m':'过去24个月开立的分期付款帐户数目','inq_last_12m':'申请日前12个月的信贷咨询次数','mths_since_recent_bc_dlq':'最近一次银行卡欠款以来的月数',\
						  'num_accts_ever_120_pd':'逾期120天或以上的帐户数','pct_tl_nvr_dlq':'百分之几的交易从未拖欠','pub_rec_bankruptcies':'公共记录破产数量',\
						  'grade':'第三方机构给出的客户评级'},inplace = True)'''
print('lending club数据导入成功\n')
pd.set_option('display.max_columns', 100)  # 显示的最大列数，如果超额就显示省略号
# print(loan_data.head(5))
print(loan_data.info(verbose=True, null_counts=True))  # 设置显示所有特征以及显示空值的参数

# 三.描述性统计分析
print('--------------------描述性统计分析--------------------\n')
# data=pd.read_csv('loan_data.csv')
pd.set_option('display.max_columns', 22)  # 显示的最大列数，如果超额就显示省略号

# 数据转换

loan_data['term'] = np.where(loan_data['term'] == ' 36 months', 36.0, 60.0)
loan_data['new_emp_length'] = loan_data['emp_length'].apply(FT.f)
loan_data.drop('emp_length', axis=1, inplace=True)
loan_data.rename(columns={'new_emp_length': 'emp_length'}, inplace=True)

# 分组汇总，计算最大值，中位数，平均值
var = ['loan_amnt', 'term', 'annual_inc', 'dti', 'delinq_2yrs', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
       'total_acc', 'acc_now_delinq', 'open_il_24m', 'inq_last_12m', 'num_accts_ever_120_pd', 'pct_tl_nvr_dlq',
       'pub_rec_bankruptcies', 'emp_length']
grouped11 = loan_data.groupby(['home_ownership'])[['loan_amnt', 'annual_inc', 'total_acc', 'dti']].agg(
    [np.max, np.median, np.mean])
print("根据'home_ownership'分组计算'loan_amnt','annual_inc','total_acc','dti'结果:\n{}".format(grouped11))
print()
grouped12 = loan_data.groupby(['home_ownership'])[['mths_since_last_delinq', 'pct_tl_nvr_dlq']].agg(
    [np.max, np.median, np.mean])
print("根据'home_ownership'分组计算结果:\n{}".format(grouped12))
print()
grouped21 = loan_data.groupby(['verification_status'])[['loan_amnt', 'annual_inc', 'total_acc', 'dti']].agg(
    [np.max, np.median, np.mean])
print("根据'verification_status'分组计算指标'loan_amnt','annual_inc','total_acc','dti'结果:\n{}".format(grouped21))
print()
grouped22 = loan_data.groupby(['verification_status'])[['mths_since_last_delinq', 'pct_tl_nvr_dlq']].agg(
    [np.max, np.median, np.mean])
print("根据'verification_status'分组计算'mths_since_last_delinq','pct_tl_nvr_dlq'结果:\n{}".format(grouped22))
print()
grouped31 = loan_data.groupby(['term'])[['loan_amnt', 'annual_inc', 'total_acc', 'dti']].agg(
    [np.max, np.median, np.mean])
print("根据'term'分组计算'loan_amnt','annual_inc','total_acc','dti'结果:\n{}".format(grouped31))
print()
grouped32 = loan_data.groupby(['term'])[['mths_since_last_delinq', 'pct_tl_nvr_dlq']].agg([np.max, np.median, np.mean])
print("根据'term'分组计算'mths_since_last_delinq','pct_tl_nvr_dlq'结果:\n{}".format(grouped32))
print()
grouped41 = loan_data.groupby(['purpose'])[['loan_amnt', 'total_acc', 'dti']].agg([np.max, np.min, np.mean])
print("根据'purpose'分组计算'loan_amnt','total_acc','dti'结果:\n{}".format(grouped41))
print()
grouped42 = loan_data.groupby(['purpose'])[['mths_since_last_delinq']].agg([np.max, np.median, np.mean])
print("根据'purpose'分组计算'mths_since_last_delinq'结果:\n{}".format(grouped42))
# 计算上下四分位数
quantile = loan_data[var].quantile([0.25, 0.75])
print("数值型数据下上四分位数为：\n{}".format(quantile))
print()
# 计算众数
mode = loan_data[['home_ownership', 'verification_status', 'term', 'purpose']].mode()
print("分类变量的众数分别为：\n'home_ownship': {} \n'verification_status'：{}\n'term'：{} \n'purpose': {}".format(
    mode['home_ownership'][0], \
    mode['verification_status'][0], mode['term'][0], mode['purpose'][0]))
print()
# 计算所有列的偏度峰度
skew = loan_data[var].skew(axis=0)
kurtosis = loan_data[var].kurtosis(axis=0)
print("数据偏度：\n{} \n\n数据峰度：\n{}".format(skew, kurtosis))

'''#计算异常值上下限
ql=loan_data.quantile(0.25)
qu=loan_data.quantile(0.75)
iqr=qu-ql
max_=qu+1.5*iqr
min_=ql-1.5*iqr
print()
print("异常值上限:\n{}\n\n异常值下限:\n{}".format(max_,min_))'''

# 四.探索性数据分析
print('--------------------探索性数据分析--------------------\n')

plt.style.use('ggplot')  # 设置绘图风格

fig1 = plt.figure('fig1', figsize=(12, 18))
ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)  # row和col各占两个
ax2 = plt.subplot2grid((4, 4), (0, 2))
ax3 = plt.subplot2grid((4, 4), (0, 3))
ax4 = plt.subplot2grid((4, 4), (1, 2))
ax5 = plt.subplot2grid((4, 4), (1, 3))
ax6 = plt.subplot2grid((4, 4), (2, 0), colspan=2)  # col占两个
ax7 = plt.subplot2grid((4, 4), (2, 2), colspan=2)  # col占两个
ax8 = plt.subplot2grid((4, 4), (3, 0), colspan=3)  # col占三个
ax9 = plt.subplot2grid((4, 4), (3, 3))

# ax1
# 生成数据集并画图
target_data = pd.value_counts(loan_data['target'], sort=True)
x1_pos = np.array(range(len(target_data.values)))
bars1 = ax1.bar(x1_pos, target_data.values, width=0.65, color='blue', alpha=0.5)
# 设置坐标轴参数
x1_labels = ['Fully Paid', 'Charged Off']
plt.setp(ax1, xticks=x1_pos, xticklabels=x1_labels)  # 多个子图设置坐标轴时用setup
plt.sca(ax1)  # 选择子图
plt.tick_params(labelsize=13)  # 设置所有xticks以及yticks的字体大小
plt.xlabel('好坏样本', fontsize=15)
plt.ylabel('数量', fontsize=15)
plt.ylim((0, target_data.max() + 20000))
plt.title('样本总体状况')
# 设置标注
for a, b in zip(x1_pos, target_data.values):
    plt.text(a - 0.1, b + 2000, b, fontsize=13)

# ax2
# 对目标列target进行清洗
loan_data['target'] = np.where(loan_data['target'] == 'Charged Off', 1, 0)
# 对工作年限进行切分
bins = [0, 3, 6, 9, 11]
x2_labels = ['3年以内', '3~6年', '6~9年', '9年以上']
loan_data['new_emp_length_cut'] = pd.cut(loan_data['emp_length'], bins=bins, labels=x2_labels, right=False)  # 新增一列工龄分段列
# print(loan_data.head())
# 构造数据集并画图
emp_length_target = loan_data.groupby('new_emp_length_cut')['target'].mean()
# print(emp_length_target)
x2_pos = np.array(range(len(loan_data['new_emp_length_cut'].unique()) - 1))  # （-1）有缺失值
bars2 = ax2.bar(x2_pos, emp_length_target, width=0.6, color='mediumorchid', alpha=0.6)
# 设置坐标轴
plt.sca(ax2)
plt.setp(ax2, xticks=x2_pos, xticklabels=x2_labels)
plt.ylim((0, emp_length_target.max() + 0.1))
plt.ylabel('坏样本占比', fontsize=11)
plt.title('工龄-坏样本', fontsize=12)
# 设置标注
for c, d in zip(x2_pos, emp_length_target):
    plt.text(c - 0.25, d + 0.02, np.round(d, decimals=3), fontsize=9)
'''#设置图案
patterns = ('x', '\\', '*', 'o', 'O', '.')
for bar,pattern in zip(bars2,patterns):
    bar.set_hatch(pattern)'''

# ax3
# 年收入描述性统计
# print(loan_data['annual_inc'].describe())
# 年收入切分
bins = [0, 30000, 60000, 90000, 120000, 99999999]
x3_labels = ['低', '较低', '中等', '较高', '高']
loan_data['new_annual_inc'] = pd.cut(loan_data['annual_inc'], bins=bins, labels=x3_labels, right=False)  # 新增一列年收入分段列
# print(loan_data.head())
# 构造数据集并画图
new_annual_target = loan_data.groupby('new_annual_inc')['target'].mean()
# print(new_annual_target)
x3_pos = np.array(range(len(loan_data['new_annual_inc'].unique())))
bars3 = ax3.bar(x3_pos, new_annual_target, width=0.6, color='yellow', alpha=0.6)
# 设置坐标轴
plt.sca(ax3)
plt.setp(ax3, xticks=x3_pos, xticklabels=x3_labels)
plt.ylim((0, new_annual_target.max() + 0.1))
plt.ylabel('坏样本占比', fontsize=11)
plt.title('年收入-坏样本', fontsize=12)
# ax3.set_xticklabels(labels=x3_labels,rotation = 20,horizontalalignment='right')
# 设置标注
for e, f in zip(x3_pos, new_annual_target):
    plt.text(e - 0.3, f + 0.02, np.round(f, decimals=3), fontsize=9)

# ax4
# 生成数据集并画图
term_data = pd.value_counts(loan_data['term'], sort=True)
x4_pos = np.array(range(len(term_data.values)))
term_target = loan_data.groupby('term')['target'].mean()
bars4 = ax4.bar(x4_pos, term_target, width=0.6, color='lightgreen', alpha=0.6)
# 设置坐标轴参数
x4_labels = ['36 months', '60 months']
plt.setp(ax4, xticks=x4_pos, xticklabels=x4_labels)  # 多个子图设置坐标轴时用setup
plt.sca(ax4)  # 选择子图
plt.ylabel('坏样本占比', fontsize=11)
plt.ylim((0, term_target.max() + 0.12))
# 设置标注
for g, h in zip(x4_pos, term_target.values):
    plt.text(g - 0.12, h + 0.02, np.round(h, decimals=3), fontsize=10)
plt.text(0.05, 0.305, '还款期限-坏样本', fontsize=12)

# ax5
# 公共不良记录描述性统计
# print(loan_data['pub_rec'].value_counts())
# 公共不良记录切分
bins = [0, 1, 2, 99]
x5_labels = ['无', '1次', '>2次']
loan_data['new_pub_rec'] = pd.cut(loan_data['pub_rec'], bins=bins, labels=x5_labels, right=False)  # 新增一列公共不良记录分段列
# 构造数据集并画图
new_pub_rec_target = loan_data.groupby('new_pub_rec')['target'].mean()
# print(new_pub_rec_target)
x5_pos = np.array(range(len(loan_data['new_pub_rec'].unique())))
bars5 = ax5.bar(x5_pos, new_pub_rec_target, width=0.6, color='pink', alpha=0.6)
# 设置坐标轴
plt.sca(ax5)
plt.setp(ax5, xticks=x5_pos, xticklabels=x5_labels)
plt.ylim((0, new_pub_rec_target.max() + 0.1))
plt.ylabel('坏样本占比', fontsize=11)
# 设置标注
for i, j in zip(x5_pos, new_pub_rec_target):
    plt.text(i - 0.2, j + 0.02, np.round(j, decimals=3), fontsize=9)
plt.text(0.12, 0.26, '公共不良记录-坏样本', fontsize=12)

# ax6
# 收入-负债比描述性统计
# print(loan_data['dti'].describe())
# 生成数据集并画图
bins = [i for i in range(0, 33, 4)]
bins.append(1000)
x6_labels = ['0~4', '4~8', '8~12', '12~16', '16~20', '20~24', '24~28', '28~32', '>32']
loan_data['new_dti'] = pd.cut(loan_data['dti'], bins=bins, labels=x6_labels, right=False)  # 新增一列(收入-负债比)分段列
# print(loan_data['new_dti'].head(10))
x = [i for i in range(0, 33, 4)]
y = loan_data.groupby('new_dti')['target'].sum() / loan_data.groupby('new_dti')['target'].count()  # 坏客户占比
plt.sca(ax6)
ax6.plot(x, y, color='b', linestyle=':', marker='o', markerfacecolor='r', markersize=6)
plt.ylabel('坏样本占比')
plt.text(2, 0.2, '负债比-坏样本', fontsize=14)

# ax7
# 生成数据集并画图
x7_pos = np.array(range(len(loan_data['new_annual_inc'].unique())))
inc_ver_target = loan_data.pivot_table(index='new_annual_inc', columns='verification_status', values='target',
                                       aggfunc='mean')
# print(inc_ver_target)
inc_ver1_target = inc_ver_target.iloc[:, 0]
inc_ver2_target = inc_ver_target.iloc[:, 1]
inc_ver3_target = inc_ver_target.iloc[:, 2]
width = 0.25
groups = ['Not Verified', 'Source Verified', 'Verified']
bars6 = ax7.bar(x7_pos, inc_ver1_target, width, alpha=0.5, color='yellow', label=groups[0])
bars7 = ax7.bar([p + width for p in x7_pos], inc_ver2_target, width, alpha=0.5, color='lightskyblue', label=groups[1])
bars8 = ax7.bar([p + width * 2 for p in x7_pos], inc_ver3_target, width, alpha=0.5, color='deeppink', label=groups[2])
# 设置坐标轴参数
x7_labels = ['低收入', '中低收入', '中等收入', '中高收入', '高收入']
plt.setp(ax7, xticks=x7_pos + 0.25, xticklabels=x7_labels)
plt.sca(ax7)
plt.ylim((0, inc_ver3_target.max() + 0.2))
plt.ylabel('坏样本占比', fontsize=10)
# 生成图例并设置位置参数
plt.legend(loc='upper center', labels=groups, bbox_to_anchor=(0.5, 1), ncol=3, framealpha=0.5)
# 设置标注
for k, l in zip(x7_pos, inc_ver1_target):
    plt.text(k - 0.14, l + 0.02, np.round(l, decimals=2), fontsize=9)
for m, n in zip([p + width for p in x7_pos], inc_ver2_target):
    plt.text(m - 0.14, n + 0.02, np.round(n, decimals=2), fontsize=9)
for o, p in zip([p + width * 2 for p in x7_pos], inc_ver3_target):
    plt.text(o - 0.14, p + 0.02, np.round(p, decimals=2), fontsize=9)

# ax8
# 构造数据集并画图
purpose_target = loan_data.groupby('purpose')['target'].mean().sort_values(ascending=False)
x8_labels = np.array(purpose_target.index)
# print(purpose_target)
x8_pos = np.array(range(len(loan_data['purpose'].unique())))
bars9 = ax8.bar(x8_pos, purpose_target, width=0.6, color='aquamarine', alpha=0.6)
# 设置坐标轴
plt.sca(ax8)
plt.setp(ax8, xticks=x8_pos, xticklabels=x8_labels)
plt.ylim((0, purpose_target.max() + 0.1))
plt.ylabel('坏样本占比', fontsize=11)
# ax3.set_xticklabels(labels=x3_labels,rotation = 20,horizontalalignment='right')
# 设置标注
for q, r in zip(x8_pos, purpose_target):
    plt.text(q - 0.25, r + 0.02, np.round(r, decimals=3), fontsize=9)
plt.text(7, 0.28, '贷款目的-坏样本', fontsize=14)
ax8.set_xticklabels(labels=x8_labels, rotation=30, horizontalalignment='right')
plt.tick_params(labelsize=12)

# ax9
# 生成数据集并画图
term1_data = np.array(loan_data.loc[loan_data['term'] == 36.0, 'loan_amnt'])
term2_data = np.array(loan_data.loc[loan_data['term'] == 60.0, 'loan_amnt'])
term_data = np.array([term1_data, term2_data])
# print(term1_data)
violinplot = ax9.violinplot(term_data, showmeans=True, showmedians=False)
# 设置坐标轴参数
x9_pos = np.array(range(len(loan_data['term'].unique())))
x9_labels = ['36 months', '60 months']
plt.setp(ax9, xticks=x9_pos + 1, xticklabels=x9_labels)
plt.sca(ax9)
plt.tick_params(labelsize=12)
plt.ylim((0, loan_data['loan_amnt'].max() + 10000))
plt.ylabel('贷款金额', fontsize=12)

# ax10(嵌套)
# 房屋拥有情况描述性统计
# print(loan_data['home_ownership'].describe())
# 设置嵌套子图参数
left, bottom, width, height = [0.31, 0.7, 0.16, 0.25]  # 方形图嵌套
ax10 = fig1.add_axes([left, bottom, width, height])
# 设置标签及画图(饼图)
explode = [0.05, 0.05, 0.05]
colors = ['gold', 'lightpink', 'floralwhite', 'mediumorchid']
# 由于any(120个)none(2个)的数量太少，无法作为是否为好样本的评判标准，创建新数据集ax10_labels，将这两个取值从在新数据集中去除
loan_data1 = loan_data.loc[(loan_data['home_ownership'] == 'MORTGAGE') | (loan_data['home_ownership'] == 'RENT') | (
        loan_data['home_ownership'] == 'OWN')]
home_ownership1_data = np.array(
    loan_data1.loc[loan_data['home_ownership'] == 'MORTGAGE', 'target'].count() / loan_data['target'].count())
home_ownership2_data = np.array(
    loan_data1.loc[loan_data['home_ownership'] == 'RENT', 'target'].count() / loan_data['target'].count())
home_ownership3_data = np.array(
    loan_data1.loc[loan_data['home_ownership'] == 'OWN', 'target'].count() / loan_data['target'].count())
home_ownership_data = np.array([home_ownership1_data, home_ownership2_data, home_ownership3_data])
labels = np.array(loan_data1['home_ownership'].unique())
ax10_labels = []
for i in labels:
    i = str(i).capitalize()
    ax10_labels.append(i)
patches, texts, autotexts = ax10.pie(home_ownership_data, labels=ax10_labels, autopct='%1.1f%%', explode=explode,
                                     colors=colors, radius=1)
# 设置图内字体大小及颜色
for text in texts + autotexts:
    text.set_fontsize(12)
for text in autotexts:
    text.set_color('black')
plt.show()

# fig2
# 生成数据集
bad = loan_data.groupby('addr_state')['target'].sum()  # 按地区分组的坏客户
good = loan_data.groupby('addr_state')['target'].count() - bad  # 按地区分组的好客户
bad_ratio = bad / (bad + good)  # 各地区坏客户占比
good.sort_values(ascending=False, inplace=True)
bad_ratio.sort_values(ascending=False, inplace=True)
# 画图
fig2 = plt.figure(figsize=(20, 8))
plt.subplot(211)  # 图1
plt.bar(good.index, good, color='cornflowerblue', alpha=0.7)
plt.bar(bad.index, bad, color='forestgreen', alpha=0.7)
plt.legend(['好样本', '坏样本'], fontsize=13)
plt.title('各个州好坏样本的数量', fontsize=16)
plt.ylabel('数量')
plt.subplot(212)  # 图2
plt.bar(bad_ratio.index, bad_ratio, color='cornflowerblue', alpha=0.7)
plt.ylabel('坏样本占比')
plt.hlines(min(bad_ratio), 50, -1, linestyle='dashed')
plt.hlines(max(bad_ratio), 50, -1, linestyle='dashed')
plt.text(30, 0.19, '各个州坏样本占比', fontsize=16)
plt.ylim((0, max(bad_ratio) + 0.03))
plt.text(-0.6, 0.237, np.round(max(bad_ratio), decimals=2), fontsize=9)
plt.text(48.6, 0.095, np.round(min(bad_ratio), decimals=2), fontsize=9)

plt.show()
# 删除新增的已切分变量，后面用卡方分箱进行切分
del loan_data['new_emp_length_cut']
del loan_data['new_annual_inc']
del loan_data['new_pub_rec']
del loan_data['new_dti']
# print(loan_data.info())
print('探索性数据分析已完成\n')

# 五.数据预处理
print('----------------------数据预处理----------------------\n')

for i in np.array(loan_data.columns):
    x = loan_data[i]
    FT.drop_duplicate(x, i)

# 2.检测并去除重复行
print('数据集是否存在重复观测: ', any(loan_data.duplicated()))
print('\n重复值处理完毕\n')

# 3.对数值型变量进行异常值检验并修改
# 统计变量类型为数值型且75%分位数>0的变量，否则IQR=0
loan_data_float_type_75 = []
loan_data_type = list(loan_data.dtypes.loc[loan_data.dtypes.values != 'object'].index)
loan_data_float_type = loan_data_type[1:]
print('去掉目标变量，其他变量类型为数值型的变量为：')
print(loan_data_float_type)

for i in loan_data_float_type:
    if loan_data[i].describe().loc['75%'] > 0:
        loan_data_float_type_75.append(i)
print('\n去掉目标变量，其他变量类型为数值型且75%分位数>0的变量为：')
print(loan_data_float_type_75)
print()

# 检测并修改(删除异常值27000个)
# 只能连续检测并修改，否则复检时会生成另外的Q1,Q3,IQR，使结果发生错误
for i in loan_data_float_type_75:
    Q1 = loan_data[i].quantile(q=0.25)
    Q3 = loan_data[i].quantile(q=0.75)
    IQR = Q3 - Q1
    check1 = any(loan_data[i] > Q3 + 1.5 * IQR)
    if check1 == True:
        print('{0}特征有高于上限的异常值'.format(i))
        drop_index1 = list(loan_data.loc[loan_data[i] > Q3 + 1.5 * IQR].index)
        loan_data.drop(drop_index1, axis=0, inplace=True)
        check1 = any(loan_data[i] > Q3 + 1.5 * IQR)
        if check1 == False:
            print('{0}特征异常值删除完毕\n'.format(i))
    else:
        print('{0}特征无高于上限的异常值\n'.format(i))

print()

for i in loan_data_float_type_75:
    Q1 = loan_data[i].quantile(q=0.25)
    Q3 = loan_data[i].quantile(q=0.75)
    IQR = Q3 - Q1
    check2 = any(loan_data[i] < Q1 - 1.5 * IQR)
    if check2 == True:
        print('{0}特征有低于下限的异常值'.format(i))
        drop_index2 = list(loan_data.loc[loan_data[i] < Q1 - 1.5 * IQR].index)
        loan_data.drop(drop_index2, axis=0, inplace=True)
        check2 = any(loan_data[i] < Q1 - 1.5 * IQR)
        if check2 == False:
            print('{0}特征异常值删除完毕\n'.format(i))
    else:
        print('{0}特征无低于下限异常值\n'.format(i))
print('异常值检测并处理完毕\n')
print()
loan_data.reset_index(level=None, drop=True, inplace=True)  # 去除异常值后重置索引列

# 4.对数据进行缺失值检验(>70%直接删除，30%~60%考虑可能是非随机缺失，可以作为一种新的分箱，<30%利用其它方法进行填充)
# 缺失值比例>70%，直接剔除
print('各列缺失比例：')
print(loan_data.apply(lambda x: sum(x.isnull() / len(x)), axis=0))
print()
columns1 = loan_data.apply(lambda x: sum(x.isnull()) / len(x), axis=0) > 0.7
na_columns = columns1[columns1.values == True].index
print('缺失值占比大于70%的特征为{}'.format(np.sum(na_columns)))  # np.sum(na_columns)可以直接将标签写出
print('缺失值占比大于70%的特征一共有{}个,将其剔除\n'.format(np.sum(columns1)))  # np.sum(columns1)可以直接将columns1中值为True的进行累加
del loan_data[np.sum(na_columns)]
# print(loan_data.info())

'''如下方法也可以剔除失值>70%的特征
for i,v in columns1.items():
	print('index:{}        values:{}'.format(i,v))
	if v == True:
		titanic.drop(labels = i,axis = 1,inplace=True)'''

# 缺失值比例介于30%~60%，考虑为非随机缺失，作为一种新的分箱('mths_since_last_delinq':'自上次违约以来的月数')(最大值为101，缺失值用999填充)
columns2 = (loan_data.apply(lambda x: sum(x.isnull()) / len(x), axis=0) > 0.3) & (
        loan_data.apply(lambda x: sum(x.isnull()) / len(x), axis=0) < 0.6)
# print(columns2)
for i, v in columns2.items():
    if v == True:
        loan_data[i] = loan_data[i].fillna(999)
print('缺失值比例介于30%~60%的特征共有1个，作为新的分箱\n')

# 检测将缺失值单独作为一种分箱的效果
loan_data['mths_since_last_delinq'] = loan_data['mths_since_last_delinq'].astype('float64')
# print(loan_data['mths_since_last_delinq'].describe(percentiles=[.25,.5,.75,.9]))
bins = [0, 25, 50, 75, 120, 1000]
labels = ['0~25', '25~50', '50~75', '75~120', '120~1000']
loan_data['mths_since_last_delinq_new'] = pd.cut(loan_data['mths_since_last_delinq'], bins=bins, labels=labels,
                                                 right=False)  # 新增一列年龄分段列
print(loan_data['mths_since_last_delinq_new'])
print(loan_data.groupby(by='mths_since_last_delinq_new')['target'].mean())

# 缺失值比例小于30%，利用众数或平均值进行填充
columns3 = (loan_data.apply(lambda x: sum(x.isnull()) / len(x), axis=0) < 0.3) & (
        loan_data.apply(lambda x: sum(x.isnull()) / len(x), axis=0) > 0)
# print(columns3)
# (1)对工作年限进行清洗、转换为数值类型并用平均值填充
'''def f(x):
    if 'years' in str(x):
        x = str(x).strip('years')
        x = str(x).replace('+','')
    elif '<' in str(x):
        x = str(x).strip('year')
        x = str(x).strip('<')
        x = float(x)-1
    else:
        x = str(x).strip('year')
    return float(x)
loan_data['emp_length']=loan_data['emp_length'].apply(f)'''
# print(loan_data['emp_length'].describe(percentiles=[.25,.5,.75]))
loan_data['emp_length'] = loan_data['emp_length'].fillna(loan_data['emp_length'].mean())
# (2)对于负债比，用均值进行填充
loan_data['dti'] = loan_data['dti'].fillna(loan_data['dti'].mean())
print('缺失值比例小于30%特征共有2个，用平均值进行填充\n')

print('缺失值处理完毕')
# print(loan_data.apply(lambda x: sum(x.isnull()/len(x)),axis=0))


# 六.特征工程
print('----------------------特征工程----------------------\n')

# 1.特征衍生
loan_data['acc_rat'] = loan_data['open_acc'] / loan_data['total_acc']  # 未结信贷产品数/总信贷产品数(新特征)
# print(loan_data['acc_rat'].describe())
# 测试新特征的可解释性
'''bins=[0,0.25,0.5,0.75,1.1]
labels=['0~0.25','0.25~0.5','0.5~0.75','0.75~1']
loan_data['new_acc_rat'] =  pd.cut(loan_data['acc_rat'], bins = bins,labels=labels,right=False) #新增一列新特征分段列
print(loan_data.groupby('new_acc_rat')['target'].mean())
del loan_data['new_acc_rat']'''
# print(loan_data.info())


# 2.卡方分箱

'''分箱原则：(1)连续型变量可以直接进行分箱
            (2)类别性变量：类别较多时：先用bad rate进行编码再用连续型分箱方式进行分箱
						 类别较少时：若每一类中均含有好坏样本则不需要分箱，否则需要将类别进行合并
						 
1、需要进行卡方分箱的连续型变量：['loan_amnt'(贷款金额),'annual_inc'(年收入),'dti'(负债比),'mths_since_last_delinq'(自上次违约以来的月数),\
							  'open_acc'(借款人未结信用产品的数目),'total_acc'(借款人当前的信用产品总数),'inq_last_12m'(申请日前12个月的信贷咨询次数),\
							  'emp_length'(工作年限),'acc_rat'(未结信贷产品比例)]     共9个特征
							  
2、需要先根据bad rate编码再进行卡方分箱的类别性变量：['purpose'(贷款目的),'addr_state'(所在州)]     共2个特征

3、不需要卡方分箱的类别性变量:['term'(还款期限),'home_ownership'(房屋拥有情况),'verification_status'(工资是否经过核实),'delinq_2yrs'(过去2年中逾期30天以上违约事件的次数),\
						'pub_rec'(公共不良记录数),'open_il_24m'(过去24个月开立的分期付款帐户数目),'num_accts_ever_120_pd'(逾期120天或以上的帐户数),\
						'pct_tl_nvr_dlq'(百分之几的交易从未拖欠),'pub_rec_bankruptcies'(公共记录破产数量)]     共9个特征'''
# print(loan_data['num_accts_ever_120_pd'].describe(percentiles=[.05,.1,.2,.3,.4,.5,.8,.85,.9,.95]))

# data=loan_data.copy()
loan_data['acc_rate'] = loan_data['open_acc'] / loan_data['total_acc']  # 未结信贷产品数/总信贷产品数(新特征)

overallRate = sum(loan_data['target']) / len(loan_data)
overallRate = round(overallRate, 6)

# type1:直接卡方分箱的连续性变量
L0 = FT.ChiMerge_MaxInterval_Original(loan_data, 'emp_length', overallRate, max_interval=5)  # L:tuple
L1 = FT.ChiMerge_MaxInterval_Original(loan_data, 'open_acc', overallRate, max_interval=2)
L2 = FT.ChiMerge_MaxInterval_Original(loan_data, 'total_acc', overallRate, max_interval=5)
L3 = FT.ChiMerge_MaxInterval_Original(loan_data, 'inq_last_12m', overallRate, max_interval=4)
loan_data.loc[loan_data['mths_since_last_delinq'] == 999, 'mths_since_last_delinq'] = np.NAN
L4 = FT.ChiMerge_MaxInterval_Original(loan_data, 'mths_since_last_delinq', overallRate, max_interval=4)
# L5=FT.ChiMerge_MaxInterval_Original(loan_data, 'acc_rat', overallRate,max_interval=5)


print("-----------------type1:bad rate 单调性检验---------------")
print("'{}':{}".format(L0[1].columns[0], FT.BadRateMonotone(L0[1], L0[1].columns[0], L0[0])))
print("'{}':{}".format(L1[1].columns[0], FT.BadRateMonotone(L1[1], L1[1].columns[0], L1[0])))
print("'{}':{}".format(L2[1].columns[0], FT.BadRateMonotone(L2[1], L2[1].columns[0], L2[0])))
print("'{}':{}".format(L3[1].columns[0], FT.BadRateMonotone(L3[1], L3[1].columns[0], L3[0])))
print("'{}':{}".format(L4[1].columns[0], FT.BadRateMonotone(L4[1], L4[1].columns[0], L4[0])))
print('检验通过,emp_length允许bad rate存在微弱的非单调性，符合实际情况，可接受')

print("----------------------type1:最终的分箱结果---------------------")
print("'{}'的分箱结果 :\n{}\n".format(L0[1].columns[0], L0[0]))
print("'{}'的分箱结果 :\n{}\n".format(L1[1].columns[0], L1[0]))
print("'{}'的分箱结果 :\n{}\n".format(L2[1].columns[0], L2[0]))
print("'{}'的分箱结果 :\n{}\n".format(L3[1].columns[0], L3[0]))
print("'{}'的分箱结果 :\n{}\n".format(L4[1].columns[0], L4[0]))

# type2:间接卡方分箱的连续性变量
# 考虑现将变量进行无监督分隔，在用有监督分隔作出权衡
# annual_inc
qcut1 = pd.qcut(loan_data['annual_inc'], 10)
loan_data['inc_level'] = loan_data['annual_inc']

loan_data.loc[(loan_data['annual_inc'] > 60000) & (loan_data['annual_inc'] <= 68000), 'inc_level'] = 5
loan_data.loc[(loan_data['annual_inc'] > 90000) & (loan_data['annual_inc'] <= 105000), 'inc_level'] = 8
loan_data.loc[(loan_data['annual_inc'] > 45000) & (loan_data['annual_inc'] <= 52000), 'inc_level'] = 3
loan_data.loc[(loan_data['annual_inc'] > 105000) & (loan_data['annual_inc'] <= 134405.4), 'inc_level'] = 9
loan_data.loc[(loan_data['annual_inc'] > 134405.4) & (loan_data['annual_inc'] <= 9522972.0), 'inc_level'] = 10
loan_data.loc[(loan_data['annual_inc'] > 68000) & (loan_data['annual_inc'] <= 78000), 'inc_level'] = 6
loan_data.loc[(loan_data['annual_inc'] > -0.001) & (loan_data['annual_inc'] <= 35000), 'inc_level'] = 1
loan_data.loc[(loan_data['annual_inc'] > 52000) & (loan_data['annual_inc'] <= 60000), 'inc_level'] = 4
loan_data.loc[(loan_data['annual_inc'] > 78000) & (loan_data['annual_inc'] <= 90000), 'inc_level'] = 7
loan_data.loc[(loan_data['annual_inc'] > 35000) & (loan_data['annual_inc'] <= 45000), 'inc_level'] = 2

inc_level = loan_data.pop('inc_level')
loan_data.insert(6, 'inc_level', inc_level)

# dti
qcut2 = pd.qcut(loan_data['dti'], 15)
loan_data['dti_type'] = loan_data['dti']

# pd.value_counts(qcut2,ascending=True)
loan_data.loc[(loan_data['dti'] > -0.001) & (loan_data['dti'] <= 5.31), 'dti_type'] = 1
loan_data.loc[(loan_data['dti'] > 5.31) & (loan_data['dti'] <= 8.01), 'dti_type'] = 2
loan_data.loc[(loan_data['dti'] > 8.01) & (loan_data['dti'] <= 10.12), 'dti_type'] = 3
loan_data.loc[(loan_data['dti'] > 10.12) & (loan_data['dti'] <= 11.89), 'dti_type'] = 4
loan_data.loc[(loan_data['dti'] > 11.89) & (loan_data['dti'] <= 13.51), 'dti_type'] = 5
loan_data.loc[(loan_data['dti'] > 13.51) & (loan_data['dti'] <= 15.04), 'dti_type'] = 6
loan_data.loc[(loan_data['dti'] > 15.04) & (loan_data['dti'] <= 16.59), 'dti_type'] = 7
loan_data.loc[(loan_data['dti'] > 16.59) & (loan_data['dti'] <= 18.17), 'dti_type'] = 8
loan_data.loc[(loan_data['dti'] > 18.17) & (loan_data['dti'] <= 19.82), 'dti_type'] = 9
loan_data.loc[(loan_data['dti'] > 19.82) & (loan_data['dti'] <= 21.54), 'dti_type'] = 10
loan_data.loc[(loan_data['dti'] > 21.54) & (loan_data['dti'] <= 23.47), 'dti_type'] = 11
loan_data.loc[(loan_data['dti'] > 23.47) & (loan_data['dti'] <= 25.6), 'dti_type'] = 12
loan_data.loc[(loan_data['dti'] > 25.6) & (loan_data['dti'] <= 28.2), 'dti_type'] = 13
loan_data.loc[(loan_data['dti'] > 28.2) & (loan_data['dti'] <= 31.67), 'dti_type'] = 14
loan_data.loc[(loan_data['dti'] > 31.67) & (loan_data['dti'] <= 999.0), 'dti_type'] = 15

dti_type = loan_data.pop('dti_type')
loan_data.insert(9, 'dti_type', dti_type)

# loan_amnt
qcut3 = pd.qcut(loan_data['loan_amnt'], 10)
loan_data['amnt_level'] = loan_data['loan_amnt']
# pd.value_counts(qcut3,ascending=True)

loan_data.loc[(loan_data['loan_amnt'] > 15000) & (loan_data['loan_amnt'] <= 17600.0), 'amnt_level'] = 7
loan_data.loc[(loan_data['loan_amnt'] > 30000) & (loan_data['loan_amnt'] <= 40000), 'amnt_level'] = 10
loan_data.loc[(loan_data['loan_amnt'] > 10000) & (loan_data['loan_amnt'] <= 12000), 'amnt_level'] = 5
loan_data.loc[(loan_data['loan_amnt'] > 17600) & (loan_data['loan_amnt'] <= 21275), 'amnt_level'] = 8
loan_data.loc[(loan_data['loan_amnt'] > 6000) & (loan_data['loan_amnt'] <= 8000), 'amnt_level'] = 3
loan_data.loc[(loan_data['loan_amnt'] > 12000) & (loan_data['loan_amnt'] <= 15000), 'amnt_level'] = 6
loan_data.loc[(loan_data['loan_amnt'] > 999.999) & (loan_data['loan_amnt'] <= 4000), 'amnt_level'] = 1
loan_data.loc[(loan_data['loan_amnt'] > 4000) & (loan_data['loan_amnt'] <= 6000), 'amnt_level'] = 2
loan_data.loc[(loan_data['loan_amnt'] > 21275) & (loan_data['loan_amnt'] <= 30000), 'amnt_level'] = 9
loan_data.loc[(loan_data['loan_amnt'] > 8000) & (loan_data['loan_amnt'] <= 10000), 'amnt_level'] = 4

amnt_level = loan_data.pop('amnt_level')
loan_data.insert(2, 'amnt_level', amnt_level)

# acc_rate
qcut4 = pd.qcut(loan_data['acc_rate'], 15)
# pd.value_counts(qcut4,ascending=True)
loan_data['acc_type'] = loan_data['acc_rate']

# pd.value_counts(qcut2,ascending=True)
loan_data.loc[(loan_data['acc_rate'] > -0.001) & (loan_data['acc_rate'] <= 0.263), 'acc_type'] = 1
loan_data.loc[(loan_data['acc_rate'] > 0.263) & (loan_data['acc_rate'] <= 0.316), 'acc_type'] = 2
loan_data.loc[(loan_data['acc_rate'] > 0.316) & (loan_data['acc_rate'] <= 0.355), 'acc_type'] = 3
loan_data.loc[(loan_data['acc_rate'] > 0.355) & (loan_data['acc_rate'] <= 0.389), 'acc_type'] = 4
loan_data.loc[(loan_data['acc_rate'] > 0.389) & (loan_data['acc_rate'] <= 0.421), 'acc_type'] = 5
loan_data.loc[(loan_data['acc_rate'] > 0.421) & (loan_data['acc_rate'] <= 0.45), 'acc_type'] = 6
loan_data.loc[(loan_data['acc_rate'] > 0.45) & (loan_data['acc_rate'] <= 0.478), 'acc_type'] = 7
loan_data.loc[(loan_data['acc_rate'] > 0.478) & (loan_data['acc_rate'] <= 0.5), 'acc_type'] = 8
loan_data.loc[(loan_data['acc_rate'] > 0.5) & (loan_data['acc_rate'] <= 0.543), 'acc_type'] = 9
loan_data.loc[(loan_data['acc_rate'] > 0.543) & (loan_data['acc_rate'] <= 0.577), 'acc_type'] = 10
loan_data.loc[(loan_data['acc_rate'] > 0.577) & (loan_data['acc_rate'] <= 0.615), 'acc_type'] = 11
loan_data.loc[(loan_data['acc_rate'] > 0.615) & (loan_data['acc_rate'] <= 0.667), 'acc_type'] = 12
loan_data.loc[(loan_data['acc_rate'] > 0.667) & (loan_data['acc_rate'] <= 0.722), 'acc_type'] = 13
loan_data.loc[(loan_data['acc_rate'] > 0.722) & (loan_data['acc_rate'] <= 0.81), 'acc_type'] = 14
loan_data.loc[(loan_data['acc_rate'] > 0.81) & (loan_data['acc_rate'] <= 1.0), 'acc_type'] = 15

acc_type = loan_data.pop('acc_type')
loan_data.insert(19, 'acc_type', acc_type)

K0 = FT.ChiMerge_MaxInterval_Original(loan_data, 'inc_level', overallRate, max_interval=5)
K1 = FT.ChiMerge_MaxInterval_Original(loan_data, 'dti_type', overallRate, max_interval=3)
K2 = FT.ChiMerge_MaxInterval_Original(loan_data, 'amnt_level', overallRate, max_interval=3)
K3 = FT.ChiMerge_MaxInterval_Original(loan_data, 'acc_type', overallRate, max_interval=5)

print("-----------------type2:bad rate 单调性检验---------------")
print("{}:{}".format(K0[1].columns[0], FT.BadRateMonotone(K0[1], K0[1].columns[0], K0[0])))
print("{}:{}".format(K1[1].columns[0], FT.BadRateMonotone(K1[1], K1[1].columns[0], K1[0])))
print("{}:{}".format(K2[1].columns[0], FT.BadRateMonotone(K2[1], K2[1].columns[0], K2[0])))
print("{}:{}".format(K3[1].columns[0], FT.BadRateMonotone(K3[1], K3[1].columns[0], K3[0])))
print('检验通过')

print("----------------------type2:最终的分箱结果---------------------")
print("'{}'的分箱结果 :\n{}\n".format(K0[1].columns[0], K0[0]))
print("'{}'的分箱结果 :\n{}\n".format(K1[1].columns[0], K1[0]))
print("'{}'的分箱结果 :\n{}\n".format(K2[1].columns[0], K2[0]))
print("'{}'的分箱结果 :\n{}\n".format(K3[1].columns[0], K3[0]))

# type3:高离散分类变量bad rate编码分箱
addr_state_group = loan_data.groupby('addr_state')['target'].mean()
addr_state_mapping = {i: j for i, j in zip(addr_state_group.index, addr_state_group.values)}
loan_data['addr_state'] = loan_data['addr_state'].map(addr_state_mapping)

purpose_group = loan_data.groupby('purpose')['target'].mean()
purpose_mapping = {i: j for i, j in zip(purpose_group.index, purpose_group.values)}
loan_data['purpose'] = loan_data['purpose'].map(purpose_mapping)

M0 = FT.ChiMerge_MaxInterval_Original(loan_data, 'purpose', overallRate, max_interval=5)
M1 = FT.ChiMerge_MaxInterval_Original(loan_data, 'addr_state', overallRate, max_interval=5)

print("-----------------type3:bad rate 单调性检验---------------")
print("{}:{}".format(M0[1].columns[0], FT.BadRateMonotone(M0[1], M0[1].columns[0], M0[0])))
print("{}:{}".format(M1[1].columns[0], FT.BadRateMonotone(M1[1], M1[1].columns[0], M1[0])))
print('检验通过')

print("----------------------type3:最终的分箱结果---------------------")
print("'{}'的分箱结果 :\n{}\n".format(M0[1].columns[0], M0[0]))
print("'{}'的分箱结果 :\n{}\n".format(M1[1].columns[0], M1[0]))

datachi2 = loan_data.copy()
# 条件改值,将原数据改为分分箱结果
L_type1 = [L0, L1, L2, L3, L4, K0, K1, K2, K3, M0, M1]
v0 = ['0~2 year', '3~5 year', '5.78 year', '6~9 year', '10 year']
v1 = ['0-3', '4-24']
v2 = ['2-4', '5-8', '9-19', '20-34', '35-52']
v3 = ['0.0', '1-5', '6', '7']
v4 = ['0~21 mths', '22~49 mths', '50~69 mths', '70~100 mths', '999']
v5 = ['level 1(1)', 'level 2(2)', 'level 3(3-7)', 'level 4(8)', 'level 5(9-10)']
v6 = ['type 1(1-13)', 'type 2(14)', 'type 3(15)']
v7 = ['level 1(1)', 'level 2(2.0-9.0)', 'level 3(10.0)', 'level 4(9)', 'level 5(10)']
v8 = ['1-2', '3-4', '5-13', '14', '15']
v9 = ['0.11-0.124', '0.1394', '0.1399-0.171', '0.202-0.225', '0.286']
v10 = ['0.0-0.103', '0.1113-0.1116', '0.114-0.121', '0.129-0.178', '0.184-0.224']
V = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
for L, v in zip(L_type1, V):
    for interval, value in zip(L[0], v):  # L[0]为初始分箱结果，v为修改分箱后的结果
        # print(interval)
        # print(interval,value)
        datachi2.loc[datachi2[L[1].columns[0]].isin(interval), L[1].columns[
            0]] = value  # L[1].columns[0]为每个变量名，
        # datachi2[L[1].columns[0]].isin(interval)返回在每个区间段为真，
        # datachi2.loc[datachi2[L[1].columns[0]].isin(interval), L[1].columns[0]]为该数据的某变量固定区间取值
datachi2['mths_since_last_delinq'].fillna(999, inplace=True)

# 数据整理,放在条件改值之后
datachi2.drop(['loan_amnt', 'annual_inc', 'dti', 'acc_rate', 'acc_rat'], axis=1, inplace=True)
datachi2.rename(
    columns={'amnt_level': 'loan_amnt', 'inc_level': 'annual_inc', 'dti_type': 'dti', 'acc_type': 'acc_rat'},
    inplace=True)  # 新变量都是在原变量基础上分箱得到。原数据分箱后无用，删除之用新变量命名之

# 3、类别数<5或取值较少且偏态分布严重的变量的分箱结果以及分箱后bad rate单调性检测
# (过去2年中逾期30天以上违约事件的次数)分箱
bins_delinq_2yrs = [0, 1, 2, 100]
labels_delinq_2yrs = ['0次', '1次', '>2次']
datachi2['delinq_2yrs'] = pd.cut(datachi2['delinq_2yrs'], bins=bins_delinq_2yrs, labels=labels_delinq_2yrs,
                                 right=False)  # delinq_2yrs列分段
# print(datachi2.groupby('delinq_2yrs')['target'].mean())
# (公共不良记录数)分箱
bins_pub_rec = [0, 1, 2, 100]
labels_pub_rec = ['0次', '1次', '>2次']
datachi2['pub_rec'] = pd.cut(datachi2['pub_rec'], bins=bins_pub_rec, labels=labels_pub_rec, right=False)  # pub_rec列分段
# print(datachi2.groupby('pub_rec')['target'].mean())
# (过去24个月开立的分期付款帐户数目)分箱
bins_open_il_24m = [0, 1, 2, 100]
labels_open_il_24m = ['0次', '1次', '>2次']
datachi2['open_il_24m'] = pd.cut(datachi2['open_il_24m'], bins=bins_open_il_24m, labels=labels_open_il_24m,
                                 right=False)  # open_il_24m列分段
# print(datachi2.groupby('open_il_24m')['target'].mean())
# (逾期120天或以上的帐户数)分箱：违约次数大于1次的样本量小于百分之5，单独分箱会造成准确性损失
bins_num_accts_ever_120_pd = [0, 1, 100]
labels_num_accts_ever_120_pd = ['无', '有']
datachi2['num_accts_ever_120_pd'] = pd.cut(datachi2['num_accts_ever_120_pd'], bins=bins_num_accts_ever_120_pd,
                                           labels=labels_num_accts_ever_120_pd, right=False)  # num_accts_ever_120_pd列分段
# print(datachi2.groupby('num_accts_ever_120_pd')['target'].mean())
# (百分之几的交易从未拖欠)分箱
bins_pct_tl_nvr_dlq = [0, 85, 95, 101]
labels_pct_tl_nvr_dlq = ['<85%未拖欠', '85%~95%未拖欠', '95%~100%未拖欠']
datachi2['pct_tl_nvr_dlq'] = pd.cut(datachi2['pct_tl_nvr_dlq'], bins=bins_pct_tl_nvr_dlq, labels=labels_pct_tl_nvr_dlq,
                                    right=False)  # pct_tl_nvr_dlq列分段
# print(datachi2.groupby('pct_tl_nvr_dlq')['target'].mean())
# (公共记录破产数量)分箱:公共记录破产次数大于1次的样本量小于百分之5，单独分箱会造成准确性损失
bins_pub_rec_bankruptcies = [0, 1, 100]
labels_pub_rec_bankruptcies = ['无', '有']
datachi2['pub_rec_bankruptcies'] = pd.cut(datachi2['pub_rec_bankruptcies'], bins=bins_pub_rec_bankruptcies,
                                          labels=labels_pub_rec_bankruptcies, right=False)  # delinq_2yrs列分段
# print(datachi2.groupby('pub_rec_bankruptcies')['target'].mean())

# 最终的数据集dataframe:datachi2
print(datachi2.head(10))

# 3.Woe编码以及各个特征的IV值计算
# Woe编码以及IV值生成的测试代码
'''IV_total=0
for j in range(len(under_samples_loan_data['open_il_24m'].unique())): #j在每一个特征的分箱中进行遍历
		total_good_num=under_samples_loan_data[under_samples_loan_data['target']==0]['target'].count()
		total_bad_num=under_samples_loan_data[under_samples_loan_data['target']==1]['target'].count()
		good_rat=(under_samples_loan_data.groupby('open_il_24m')['target'].count().iloc[j]-under_samples_loan_data.groupby('open_il_24m')['target'].sum().iloc[j])/total_good_num
		bad_rat=under_samples_loan_data.groupby('open_il_24m')['target'].sum().iloc[j]/total_bad_num
		Woe=float(np.log((good_rat)/(bad_rat))) 
		IV=(good_rat-bad_rat)*Woe
		IV_total+=IV
		print('total_good_num={}'.format(total_good_num))
		print('total_bad_num={}'.format(total_bad_num))
		print('good_rat={}'.format(good_rat))
		print('bad_rat={}'.format(bad_rat))
		print('Woe={}'.format(Woe))
		print('IV={}'.format(IV))
		print()'''

# 由于any(120个)none(2个)的数量太少，无法作为是否为好样本的评判标准，从数据集中删除
datachi2 = datachi2.loc[(datachi2['home_ownership'] == 'MORTGAGE') | (datachi2['home_ownership'] == 'RENT') | (
        datachi2['home_ownership'] == 'OWN')]
datachi2.reset_index(level=None, drop=True, inplace=True)
assist_data = datachi2.copy()  # 生成辅助数据集
# print(datachi2.head(20))

# 由于经过数据预处理剔除异常值后以后'term'变量只剩下一种情况(term=36)，所以将该变量从数据集中剔除
var_IV = []
var = list(datachi2.columns)  # 所有经过分箱后的特征存储
del var[0]  # 暂时剔除目标变量，不对其进行编码
del var[1]  # 删除'term'特征
# print(var)
for i in var:  # i在所有的特征中遍历
    IV_total = 0
    var_box = datachi2.groupby(i)['target'].count().index
    Woe_list = []
    for j in range(len(datachi2[i].unique())):  # j在每一个特征的分箱中进行遍历
        total_good_num = datachi2[datachi2['target'] == 0]['target'].count()
        total_bad_num = datachi2[datachi2['target'] == 1]['target'].count()
        good_rat = (datachi2.groupby(i)['target'].count().iloc[j] - datachi2.groupby(i)['target'].sum().iloc[
            j]) / total_good_num
        bad_rat = datachi2.groupby(i)['target'].sum().iloc[j] / total_bad_num
        Woe = float(np.log((good_rat) / (bad_rat)))
        IV = (good_rat - bad_rat) * Woe
        IV_total += IV
        Woe_list.append(Woe)
        print('变量{0}的对应分箱{1}的Woe编码为{2}'.format(i, j, Woe))
    var_IV.append(IV_total)
    print('变量{0}的IV值为{1}'.format(i, IV_total))
    print()
    var_mapping = {m: n for m, n in zip(var_box, Woe_list)}
    datachi2[i] = datachi2[i].map(var_mapping)  # 将所有变量的值换为其woe值
print(var_IV)
print(datachi2.head(10))
print('\nWoe编码完成')

# 各个特征IV值可视化
fig3, ax = plt.subplots(figsize=(10, 10))
# 创建‘特征—IV’字典并按照IV值降序排序
bar_labels = var  # 各特征的储存列表
var_IV = var_IV  # 各特征IV值的储存列表
dict_IV = dict(zip(bar_labels, var_IV))
dict_IV_sort = sorted(dict_IV.items(), key=lambda x: x[1], reverse=True)  # 返回的是一个嵌套列表，降序
# 生成数据集和标签并画图
keys = []
values = []
for i in range(len(dict_IV_sort)):
    keys_i = dict_IV_sort[i][0]
    values_i = dict_IV_sort[i][1]
    keys.append(keys_i)
    values.append(values_i)
x_pos = np.array(range(len(keys)))
# print(keys)
# print()
# print(values)
plt.bar(x_pos, values, color='aquamarine', alpha=0.7)
plt.ylim([0, max(var_IV) * 1.2])
plt.ylabel('各特征的IV')
plt.setp(ax, xticks=x_pos, xticklabels=keys)
ax.set_xticklabels(labels=keys, rotation=40, horizontalalignment='right')
# 生成注释
for A, B in zip(x_pos, values):
    plt.text(A - 0.41, B + 0.003, np.round(B, decimals=4), fontsize=9)
plt.hlines(0.02, 19.1, -0.5, linestyle='dashed')
plt.text(19.15, 0.017, '0.02', fontsize=11)
plt.show()

# 4.单变量分析

print('\n分箱后bad rate单调性检测完成\n')
# 选取IV值>0.02的变量
retain_var = []
for i, j in dict_IV.items():
    if j >= 0.02:
        retain_var.append(i)
print('根据IV值的筛选保留下来的变量为：')
print(retain_var)
print()
# 生成新数据集
retain_dataset = datachi2[retain_var]
# print(retain_dataset.head(9))
print()

# 5.多变量分析
# 相关矩阵图
retain_dataset_corr = retain_dataset.corr()
fig4, ax = plt.subplots(figsize=(10, 10))
# robust=True自动设置颜色,annot显示网格数据,fmt小数保留位数
ax = sns.heatmap(retain_dataset_corr, linewidths=0.5, vmax=1.2, vmin=-1.2, cmap='rainbow', annot=True, fmt='.2f')
plt.show()
# 我们剔除相关系数>0.5的变量(进行IV值大小比较后)


# 6.多重共线性分析

VIF = pd.DataFrame()
VIF["features"] = retain_dataset.columns
VIF["VIF Factor"] = [variance_inflation_factor(retain_dataset.values, i) for i in range(retain_dataset.shape[1])]
print(VIF)
print('\n多重共线性检验通过\n')

print('最终数据集：')
retain_var.insert(0, 'target')  # 加入'target'目标列（此步骤可以放在所有特征挑选完毕之后）
final_var = retain_var  # 最终变量集
final_dataset = datachi2[final_var]  # 最终数据集(下采样前)
print(final_dataset.info())
# print(final_dataset.head(10))


# 下采样
# 样本量充足，使用下采样方法处理目标样本不均衡问题
bad_num = final_dataset['target'].value_counts().loc[1]
bad_index = final_dataset.loc[final_dataset['target'] == 1].index
good_index = final_dataset.loc[final_dataset['target'] == 0].index
random_good_index = np.random.choice(good_index, bad_num, replace=True)  # 从good_index中，按照bad_num的shape提取样本
under_samples_index = np.concatenate([random_good_index, bad_index])
assist_dataset = assist_data[final_var].iloc[under_samples_index, :]  # 生成辅助评分的DataFrame
final_dataset = final_dataset.iloc[under_samples_index, :]
# print(final_dataset.info())
assist_dataset.reset_index(level=None, drop=True, inplace=True)
final_dataset.reset_index(level=None, drop=True, inplace=True)
# print(assist_dataset['target'].value_counts())
# print(assist_dataset.head())
print('\n经过下采样后好样本与坏样本的数目为：')
print(final_dataset['target'].value_counts())
# print(final_dataset.head())
final_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\风控\final_dataset.csv', index=False)  # 最终数据集
assist_dataset.to_csv(r'E:\PycharmProjects\sklearn\Data\风控\assist_dataset.csv', index=False)  # 最终辅助数据集

print("DONE")
