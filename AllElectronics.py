from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv


def trainDicisionTree(csvfileurl):
    """
    函数说明：创建决策树进行训练，然后进行测试。
    :param csvfileurl: csv文件路径
    :return:
    """
    # 创建存放特征值和目标值的列表
    featureList = []
    labelList = []

    # 读取文件信息
    allElectronicsData = open(csvfileurl)   # 打开文件
    reader = csv.reader(allElectronicsData)     # 逐行读取数据
    headers = next(reader)      # 读取信息头文件
    print('headers:\n', headers)

    # 存储特征值和目标值
    for row in reader:
        labelList.append(row[-1])   # 读取文件的最后一列存放到目标值列表中
        rowDict = {}    # 存放特征值的字典
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]    # 将特征值存放到字典中
        featureList.append(rowDict)     # 将特征值字典存放到列表中
    print('featureList:\n', featureList)
    print('labelList:\n', labelList)

    # 特征值数值化
    vec = DictVectorizer()      # 整型数字转化，将特征与值的映射字典组成的列表转换成向量
    x = vec.fit_transform(featureList).toarray()   # 将特征值转换成整型数字数组
    print('训练集特征值为:\n' + str(x))
    print(vec.get_feature_names())      # 每种特征值包含的种类

    # 目标值数值化
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform((labelList))      # 目标值标准化、归一化（均值为0，方差为1）
    print('训练集目标值为:\n' + str(y))

    # 使用决策树进行分类预测处理
    clf = tree.DecisionTreeClassifier(criterion='entropy')  # 通过信息增益构建决策树
    clf = clf.fit(x,y)
    # print('clf:\n' + str(clf))

    # 可视化决策树
    with open('allElectronicInfomationGainOri.dot', 'w') as f:
        f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(),out_file=f)

    # 创建测试集
    test = []
    oneRowX = x[0,:]   # 取训练集第一行数据
    newRowx = oneRowX
    newRowx[0] = 1
    newRowx[2] = 0
    test.append(newRowx)
    print('测试数据为:\n', test)

    # 预测数据
    pred = clf.predict(test)
    print('预测值为:', pred)



if __name__ == '__main__':
    trainDicisionTree('data/AllElectronics.csv')