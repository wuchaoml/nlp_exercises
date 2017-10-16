import os
import jieba
import random
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np 
import pylab as pl 
import matplotlib.pyplot as plt 

# 文本处理，样本的生成过程
def text_processing(folder_path,test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 遍历文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        # 读取文件
        for file in files:
            with open(os.path.join(new_folder_path,file),'r') as fp:
                raw = fp.read()
            raw = raw.strip()
            # jieba并行模式
            jieba.enable_parallel(4)
            word_list = jieba.lcut(raw)
            jieba.disable_parallel()

            data_list.append(word_list)
            class_list.append(folder)

    # 简单的划分训练集和测试集
    data_class_list = list(zip(data_list,class_list))
    # 对列表进行随机打散
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_class_list = list(zip(*train_list))
    test_data_list,test_class_list = list(zip(*test_list))

    # 统计所有单词的词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            all_words_dict.setdefault(word,0)
            all_words_dict[word] += 1

    # 根据词频对单词进行降序排列
    all_words_tuple_list = sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
    all_words_list = list(list(zip(*all_words_tuple_list))[0])

    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list

# 单词去重
def make_word_set(words_file):
    words_set = set()
    with open(words_file,'r') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set:
                words_set.add(word)
    return words_set

# 特征词的选取（1000）
def words_dict(all_words_list,deleteN,stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n>1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n+=1
    return feature_words

# 文本特征的处理
def text_features(train_data_list,test_data_list,feature_words,flag='nltk'):
    def text_features(text,feature_words):
        text_words = set(text)
        if flag == 'nltk':
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        return features
    train_feature_list = [text_features(text,feature_words) for text in train_data_list]
    test_feature_list = [text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list

# 分类，同时输出准确率
def text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag):
    if flag == 'nltk':
        train_flist = zip(train_feature_list,train_class_list)
        test_flist = zip(test_feature_list,test_class_list)
        classifier = nltk.classify.accuracy(classifier,test_flist)
    elif flag == 'sklearn':
        classifier = MultinomialNB().fit(train_feature_list,train_class_list)
        test_accuracy = classifier.score(test_feature_list,test_class_list)
    else:
        test_accuracy = []
    return test_accuracy

def main():
    # 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list,train_data_list,test_data_list,train_class_list,test_class_list = text_processing(folder_path,test_size = 0.2)

    # 停止词集合
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = make_word_set(stopwords_file)

    # 文本的特征提取和分类
    flag = 'sklearn'
    deleteNs = range(0,1000,20)
    test_accuracy_list = []
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list,deleteN,stopwords_set)
        train_feature_list,test_feature_list = text_features(train_data_list,test_data_list,feature_words,flag)
        test_accuracy = text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag)
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)

    # 结果分析
    plt.plot(deleteNs,test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()





if __name__ == '__main__':
    main()
