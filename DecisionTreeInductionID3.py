'''
Created on October 20, 2021

@author: Syed.Ausaf.Hussain
'''

import pandas
import math

def calculateEntropy(classes):
    infoD = 0 #entropy of D
    for c in classes.values():
        p = c/D
        infoD += -p * math.log2(p)
    return infoD

def calculateGain(attributes, data, infoD):
    infoDAttr = {}
    for attr in attributes[:-1]:
        infoDAttr[attr] = 0
        for uniqueValue in data[attr].unique():
            temp = data[[attr, attributes[-1]]].where(data[attr] == uniqueValue)
            dictClass = temp.groupby(attributes[-1]).count().to_dict()
            totalCount = temp.count().values[0]
            infoD_C = 0
            for key, classCount in dictClass[attr].items():
                p = classCount / totalCount
                infoD_C += -p * math.log2(p)

            infoDAttr[attr] += totalCount / D * infoD_C
        infoDAttr[attr] = infoD - infoDAttr[attr]
    return infoDAttr

def spiliting(attributes, data, node, unique, spilitAttr, infoD):
    attr = [item for item in attributes if item not in spilitAttr]
    nData = data[data[node] == unique]
    if (nData[attributes[-1]].nunique() == 1):
        spilitAttr.append('('+node+')' + unique + "->" + str(nData[attributes[-1]].unique()) + "class(Leaf node)")
    else:
        nfoDAttr = calculateGain(attr, nData, infoD)
        spilitAttr.append('('+node+')' + unique + '->' + max(nfoDAttr, key=nfoDAttr.get))
        nNode = max(nfoDAttr, key=nfoDAttr.get)
        for unique in nData[nNode].unique():
            spilitAttr = spiliting(attr, nData, nNode, unique, spilitAttr, infoD)
    return spilitAttr


data = pandas.read_csv('dataset1.csv')
print("Data\n", data)
attributes = data.columns.values.tolist()  # class attribute is the last column
classes = data[attributes[-1]].value_counts(ascending=True).to_dict()
D = data.shape[0]  # #of tuples in data
infoD = calculateEntropy(classes)
print("classes", classes)
print("Total tuples", data.shape[0])
print("infoD (entropy)", infoD)

spilitAttr = []
infoDAttr = calculateGain(attributes, data, infoD)  # for root selection
spilitAttr.append(max(infoDAttr, key=infoDAttr.get))
rootNode = max(infoDAttr, key=infoDAttr.get)
for unique in data[rootNode].unique():
    spilitAttr = spiliting(attributes, data, rootNode, unique, spilitAttr, infoD)

print("spilitAttr", spilitAttr)

