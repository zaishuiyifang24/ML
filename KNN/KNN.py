#!/usr/bin/env python
# coding: utf-8

# ## K-近邻算法笔记
# 1.  KNN算法的原理 
#     - 有监督学习，训练集中的数据是有标签的
#     - 在没有标签的数据输入之后，提取样本集中最相似数据（最近邻）的分类标签
#     - 一般只选择前k个最相似的数据
#     - 最后，选择k个最相似数据出现次数最多的分类作为最终的分类结果
# 2. 函数笔记
#     - **tile(A,reps)** A和reps都是array_like，A表示要重复的数据，reps的第一个数表示要重复的行数，第二个数表示要重复的列数。示例：diffMat = tile(inx,(dataSetSize,1)) - dataSet
#     - **sum()运算**。没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加。示例:sqDiffMat = sqDiffMat.sum(axis=1)
#     - **argsort函数**。argsort函数返回的是数组值从小到大的索引值，即排序之后的每个位置的索引。示例：sortedDistIndicies = distances.argsort()
#     - **classCount.get(voteIlabel,0)**返回字典classCount中voteIlabel元素对应的值,若无，则进行初始化。  
#         若不存在voteIlabel，则字典classCount中生成voteIlabel元素，并使其对应的数字为0，即classCount = {voteIlabel：0}
#     此时classCount.get(voteIlabel,0)作用是检测并生成新元素，括号中的0只用作初始化，之后再无作用  
#         当字典中有voteIlabel元素时，classCount.get(voteIlabel,0)作用是返回该元素对应的值，即0
#     - **items函数**，将一个字典以列表的形式返回，因为字典是无序的，所以返回的列表也是无序的。示例:a.items()
#     - **operator.itemgetter函数**。operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号。要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。示例 b=operator.itemgetter(1) 。
#     - **sort(输入数据，key=排序的关键字，reverse是否逆序)**，第一个为数据的数据，key为要排序的关键字是哪一个，reverse表示是否需要逆序
#     - 

# In[4]:


from numpy import *
import operator   #导入运算符模块
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


# In[14]:


group,labels = createDataSet()


# In[23]:


def classisy0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat  = diffMat** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount ={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


# In[26]:


classisy0([1,1],group,labels,3)

