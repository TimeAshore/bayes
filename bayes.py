"""先验为多项式分布的朴素贝叶斯"""
import jieba
import numpy as np
import time

from collections import Counter
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from single_test import unknown_text1, unknown_text2, unknown_text3, unknown_text4, unknown_text5, unknown_text6, \
    unknown_text7, unknown_text8, unknown_text9, unknown_text10, unknown_text11
from handling_data import getdata, cut_words, remove_stopwords


# 训练分类器
def start_NB(train):
    words, the_type_doc = cut_words(train)  # 分词
    words = set(words)  # 词袋模型
    print("词袋模型大小为：", len(words))

    # 样本矩阵, 样本标签
    marx, labels = [], []
    with open('traindatas.txt', 'w') as fp:
        for doc in the_type_doc:
            if len(str(doc[1]).strip("[]")) == 0:  # 舍弃未分类的样本
                continue

            # 保存训练集样本，便于查看样本的好坏
            fp.write(str(doc[0]))
            fp.write(str(doc[1]).strip("[]"))
            fp.write(str(doc[2]))
            fp.write('\n')

            labels.append(str(doc[1]).strip("[]"))
            temp_list = []  # 每个样本矩阵，按tf存，不按（0,1）
            words_tf = dict(Counter(doc[2]))
            for word in words:
                # temp_list.append(1 if word in doc[2] else 0)
                temp_list.append(words_tf[word] if word in doc[2] else 0)
            # print(temp_list)
            marx.append(temp_list)
    print("矩阵维度：　", len(marx[0]))
    X = np.array(marx)
    Y = np.array(labels)

    t1 = time.time()
    clf = MultinomialNB(alpha=1)  # 0.8 | 1.1
    clf.fit(X, Y)  # 拟合。
    # ================拟合过程===================
    #        X                           Y
    # [[10, 20, 0, ..., 0],             edu
    #  [10, 30, 10, ..., 0],            edu
    #  [0, 0, 0, ..., 30],              medic
    #  ...                              ...
    #  [0, 0, 15, ..., 0]  ]            gov
    #         ||               ||
    #         ||               ||
    #         \/               \/
    # 词对类型的类条件概率
    # edu   (0.03, 0.11, 0.02, 0.0015, 0, 0, 0.09, 0, 0, ..., 0.07, 0.000001)
    # gov   (0, 0, 0, 0, 0.007, 0.009, 0, 0.12, 0, ..., 0, 0.005)
    # medic (0, 0, 0, 0, 0, 0, 0, 0, 0.15, ..., 0, 0)
    # ==========================================

    print("拟合数据用时：", time.time() - t1)

    # 保存模型,词袋
    num = 0
    with open('words.txt', 'w') as fp:
        for word in words:
            fp.write(word)
            num += 1
            if num < len(words):
                fp.write(',')
    joblib.dump(clf, 'clf.pkl')


# 预测
def pre(clf, text, words):
    # 样本预处理
    segs = jieba.cut(text, cut_all=False)  # 分词
    html_word = list(segs)
    html_word = remove_stopwords(html_word)  # 去停用词

    # #可用词汇少于5个，跳过这个样本
    # if len(html_word) < 5 or len(text[-1]) < 1:
    #     print("【预测】可用词汇少于5个或者样本未分类，跳过这个样本", html_word)
    #     return 1

    # 统计词频
    text_tf = dict(Counter(html_word))

    # 未知文本空间向量，按tf存
    temp_list = []
    for word in words:
        # temp_list.append(1 if word in html_word else 0)
        temp_list.append(text_tf[word] if word in html_word else 0)

    # 预测
    predict_type = clf.predict([temp_list])
    # print("未知文本的预测结果是: ", predict_type)
    gl = clf.predict_proba([temp_list]).tolist()[0]  # 在各个类别上预测的概率
    # print(clf.classes_)  # 所有类别

    # 类别概率估计排序
    answer_dict = []
    for i in range(len(clf.classes_)):
        answer_dict.append((clf.classes_[i], gl[i]))
    answer_dict = sorted(answer_dict, key=lambda x: x[1], reverse=True)
    print(answer_dict)
    return predict_type  # 预测出来的类别


# 测试
def single():
    # 加载分类模型
    clf = joblib.load('clf.pkl')  # 读
    with open('words.txt', 'r') as fp:
        content = fp.read()
        words = content.split(',')

    req_list = [unknown_text1, unknown_text2, unknown_text3, unknown_text4, unknown_text5, unknown_text6, unknown_text7,
                unknown_text8, unknown_text9,
                unknown_text10, unknown_text11]

    for x in req_list:
        pre(clf, x, words)  # 预测


# 模型评估
def singless2(test_data):
    # 加载预测分类所需文件
    with open('words.txt', 'r') as fp:
        content = fp.read()
        words = content.split(',')
    clf = joblib.load('clf.pkl')

    true_y, y_pred = [], []
    dataset = []
    for text in test_data:
        segs = jieba.cut(text[0], cut_all=False)
        html_word = list(segs)
        html_word = remove_stopwords(html_word)

        # 可用词汇少于2个，跳过这个样本
        if len(html_word) < 2 or len(text[-1]) < 1:
            print("【singless2】可用词汇少于2个或者待预测样本没有真实分类，跳过这个样本", html_word)
            continue

        with open('testdatas.txt', 'a') as fp:
            # print("text[-1][0]", text[-1])
            fp.write(text[-1])
            fp.write(str(html_word))
            fp.write('\n')

        text_tf = dict(Counter(html_word))  # 统计词频
        temp_list = []  # 未知文本空间向量，按tf存
        for word in words:
            # temp_list.append(1 if word in html_word else 0)
            temp_list.append(text_tf[word] if word in html_word else 0)
        dataset.append(temp_list)
        true_y.append(text[-1])

        # 获取预测类别y_pred
        typeeeeee = pre(clf, text[0], words)
        y_pred.append(typeeeeee[0].strip("'"))

    print("测试样本数量：", len(dataset), len(true_y))
    print("真实：", len(true_y), true_y)
    print("预测：", len(y_pred), y_pred)
    dataset = np.array(dataset)  # 转化为矩阵
    true_y = np.array(true_y)

    # score = clf.score(dataset, true_y)
    # print("在（dataset,true_y）上预测的得分: ", score)

    # *********************metrics里面的方法*****************************
    confusion_mat = confusion_matrix(true_y, y_pred)
    print("混淆矩阵: ", confusion_mat)  # 混淆矩阵
    # *********************直接出性能报告， （精准率、召回率、F1值）******************
    print('*' * 50)
    print('性能报告: ')
    print(classification_report(true_y, y_pred))


if __name__ == '__main__':
    t1 = time.time()

    website_num = 20000  # 指定数据量

    rows = getdata(website_num)  # 获取数据
    for x in rows:
        print(x[1])
    rate = int(len(rows) / 5 * 4)
    train_data, test_data = rows[:], rows[rate:]

    start_NB(train_data)  # 训练

    # single()

    singless2(test_data)  # 准确率
    print("总用时：", time.time() - t1)
