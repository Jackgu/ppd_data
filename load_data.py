#coding=utf-8
import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import datetime
import matplotlib


##### 这个用于加载ppd提供下载的数据，主要工作：读取文件；对中文进行处理；去掉无效数据；类型操作；展示基本数据 #####

################################################################################################################
def LoadData():
    base_dir = os.getcwd()
    files = os.listdir(base_dir)
    for f in files:
        if f.endswith(".csv"):
            file = os.getcwd()+"/"+ f
            data = pd.read_csv(file, header=0, encoding='gb2312')
            break #only load the first file, can add feature to load & combine more data files later


    ### Remove unuseful data
    del data["recorddate"]
    #data.drop_duplicates(subset=['ListingId'], keep='first', inplace=True)  #对dataframe进行去重
    del data["ListingId"]


    ### Remove data that will have bad impact on modeling.
    data = data[data[u"借款成功日期"] <= "2016/10"]
    data = data[data[u"初始评级"].isin(["B","C","D","E","F"])]
    #data = data[data[u"初始评级"] != "AAA"]
    data = data[-data[u"借款类型"].isin([u"电商",u"其他"])]
    data = data[(data[u"借款金额"] <= 30000) & (data[u"借款金额"] > 600)] #&(data[u"借款期限"] <= 12)]
    data = data[data[u"借款利率"] > 14]


    ### Repalce Chinese Character to Number
    data = data.replace(u'男', 1)\
        .replace(u'女', 0)\
        .replace(u'是', 1)\
        .replace(u'否', 0)\
        .replace(u'成功认证', '1')\
        .replace(u'未成功认证', '0')\
        .replace(u'普通', 0)\
        .replace(u'APP闪电', 1)\
        .replace(u'其他', 2)\
        .replace(u'电商', 4)\
        .replace(u'正常还款中', 0)\
        .replace(u'逾期中', 1)\
        .replace(u'已还清', 2)


    ### Set the correct data type, avoid "object" type
    #data[u"借款成功日期"] = data[u"借款成功日期"].astype("datetime64")
    data[u"借款成功日期"] = pd.to_datetime(data[u"借款成功日期"])   #下次计划还款日期, 上次还款日期

    data[u"借款金额"] = data[u"借款金额"].astype("int")
    data[u"借款期限"] = data[u"借款期限"].astype("int")
    data[u"借款利率"] = data[u"借款利率"].astype("int")
    data[u"初始评级"] = data[u"初始评级"].astype("category")
    data[u"是否首标"] = data[u"是否首标"].astype("category")
    data[u"性别"] = data[u"性别"].astype("category")
    data[u"手机认证"] = data[u"手机认证"].astype("category")
    data[u"户口认证"] = data[u"户口认证"].astype("category")
    data[u"视频认证"] = data[u"视频认证"].astype("category")
    data[u"学历认证"] = data[u"学历认证"].astype("category")
    data[u"征信认证"] = data[u"征信认证"].astype("category")
    data[u"淘宝认证"] = data[u"淘宝认证"].astype("category")
    data[u"借款类型"] = data[u"借款类型"].astype("category")
    data[u"标当前状态"] = data[u"标当前状态"].astype("category")

    ### Calculate Default
    data["IsDefault"] = data.apply(lambda x: 1 if (x[u'标当前逾期天数'] > 1) else 0, axis=1)

    return data


################################################################################################################
### 输出一些统计信息，可以用于基本数据情况的了解
################################################################################################################
def PrintDatasetInfo(d):
    print (d.shape)
    print (d.dtypes)
    pd.set_option('display.width', 100)
    pd.set_option('precision', 3)
    print(d.describe())

    #print(d.corr(method='pearson'))
    #print(d.skew())

    print ("===================")


################################################################################################################
### 对数据进行展示，类似于"拍分析"。加上图即可。 Pandas和SQL可以类比
################################################################################################################
def PrintBusinessInfo(d):
    print
    print (u"对数据进行展示，类似于/拍分析/。加上图即可")
    ##TODO: 只是简单的处理，后续可以继续改进

    d["Y"] = d[u"借款成功日期"].dt.year
    d["Q"] = d[u"借款成功日期"].dt.quarter

    ## Should have better solution
    d1 = d.sort_values(by=u"借款成功日期").loc[:,u"借款成功日期"]
    startDate = datetime.datetime.utcfromtimestamp(d1.head(1).values[0].tolist() / 1e9).strftime('%Y/%m/%d')
    endDate = datetime.datetime.utcfromtimestamp(d1.tail(1).values[0].tolist() / 1e9).strftime('%Y/%m/%d')
    print "From {} to {}".format(startDate,endDate)    # min(CreationDate)

    ### Basic data
    print "Total Amount:{}, Count:{}".format(d[u"我的投资金额"].sum(),d[u"我的投资金额"].count())     # sum(bidamount)
    group_grade = d.groupby(u"初始评级")
    print group_grade.size()   # Group By Grade
    print group_grade.agg({u"我的投资金额":[np.size, np.sum,np.mean], u"借款利率":np.mean, "IsDefault":np.sum})

    group_grade_q = d.groupby(["Y","Q"])   # Group BY Grade/Season
    print group_grade_q.agg({u"我的投资金额": [np.size, np.sum, np.mean], u"借款利率": np.mean, "IsDefault": np.sum})

    #plt.show()

    print ""
    print ""   # sum(bidamount), average(bidamount), count(bidamount) Group BY Grade/Season
    print ""    # sum(default)/sum() Group By Grade/Season


# d = LoadData()
# Print Test data
# print d.head(2)

# Print Data information
# PrintDatasetInfo(d)

#PrintBusinessInfo(d)