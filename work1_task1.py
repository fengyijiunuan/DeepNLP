import os
import jieba
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_stopwords(file_path):
    """
    加载停用词文件。

    参数:
    file_path (str): 停用词文件的路径。

    返回:
    list: 停用词列表。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

def remove_stopwords(text, stopwords):
    """
    从文本中移除停用词。

    参数:
    text (str): 输入文本。
    stopwords (list): 停用词列表。

    返回:
    str: 移除停用词后的文本。
    """
    words = text.split()
    filtered_text = [word for word in words if word not in stopwords]
    return ' '.join(filtered_text)

def read_texts_from_directory(directory_path, encoding='gb18030'):
    """
    从指定目录读取所有文本文件并合并为一个字符串。

    参数:
    directory_path (str): 文本文件所在目录的路径。
    encoding (str): 读取文件时使用的编码，默认为'gb18030'。

    返回:
    str: 合并后的文本。
    """
    combined_text = ""
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r', encoding=encoding) as file:
            combined_text += file.read()
    return combined_text

def main(text):
    """
    主函数：执行文本预处理、分词、词频统计和Zipf定律验证。

    参数:
    text (str): 输入文本。
    """
    stopwords_file_path = r'C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\1\cn_stopwords.txt'
    cn_stopwords = load_stopwords(stopwords_file_path)
    text = remove_stopwords(text, cn_stopwords)
    irrelevant_texts = [
        '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
        '新语丝电子文库(www.xys.org)',
        '新语丝电子文库'
    ]
    for irrelevant_text in irrelevant_texts:
        text = text.replace(irrelevant_text, '')

    words = list(jieba.cut(text))
    word_freq = Counter(words)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    freq = [item[1] for item in sorted_word_freq]
    rank = np.arange(1, len(freq) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(np.log(rank), np.log(freq), marker='o', linestyle='')
    plt.title("齐夫定律验证")
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.grid(True)
    # 汉字字体，优先使用楷体，找不到则使用黑体
    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()



# 读取文本文件夹路径作为参数传递
text_directory_path = r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\jyxstxtqj_downcc.com"
combined_text = read_texts_from_directory(text_directory_path)
main(combined_text)