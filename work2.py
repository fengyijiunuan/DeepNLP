import gensim
import os
import jieba
import numpy as np
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# 文本数据预处理，去掉无关信息、停用词等
def preprocess_text(text, stop_words_list):
    text = text.replace('----〖新语丝电子文库(www.xys.org)〗', '')
    text = text.replace('本书来自www.cr173.com免费txt小说下载站', '')
    text = text.replace('更多更新免费电子书请关注www.cr173.com', '')
    text = text.replace('\u3000', '')
    words1 = jieba.lcut(text)
    jieba_para = [word for word in words1 if word not in stop_words_list and word != ' ']
    words2 = [char for char in text]
    char_para = [word for word in words2 if word not in stop_words_list and word != ' ']
    return jieba_para, char_para


def extract_jieba_paragraphs(para_num, token_num, book_para_jieba):
    corpus = []
    src_labels = []
    for file_name, paragraphs in book_para_jieba.items():
        curr = []
        for words in paragraphs:
            curr.extend(words)
            # 确保每个段落能够达到K个token
            if len(curr) < token_num:
                continue
            else:
                corpus.append(curr[:token_num])
                src_labels.append(file_name)
                curr = []

    # 将1000个段落均匀分配给每个文本
    dataset = []
    sampled_labels = []
    para_num_per_book = int(para_num / len(book_para_jieba)) + 1
    for label, label_id in book_names_id.items():
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, src_labels) if paragraph_label == label]

        # # 某些文本段落数不够，复制段落数量
        # if len(label_paragraphs) < para_num_per_book:
        #     label_paragraphs = label_paragraphs * int(para_num_per_book / len(label_paragraphs) + 1)
        sampled_index_list = np.random.choice(len(label_paragraphs), para_num_per_book, replace=True)
        sampled_paragraphs = [label_paragraphs[index] for index in sampled_index_list]

        dataset.extend(sampled_paragraphs)

        sampled_labels.extend([label_id] * para_num_per_book)

    dataset = dataset[:para_num]
    sampled_labels = sampled_labels[:para_num]

    return dataset, sampled_labels

def extract_char_paragraphs(para_num, token_num, book_para_char):
    corpus = []
    src_labels = []
    for file_name, paragraphs in book_para_char.items():
        curr = []
        for char_list in paragraphs:
            curr.extend(char_list)
            if len(curr) < token_num:
                continue
            else:
                corpus.append(curr[:token_num])
                src_labels.append(file_name)
                curr = []

    dataset = []
    sampled_labels = []
    para_num_per_book = int(para_num / len(book_para_char)) + 1
    for label, label_id in book_names_id.items():
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, src_labels) if paragraph_label == label]
        if len(label_paragraphs) < para_num_per_book:
            label_paragraphs = label_paragraphs * int(para_num_per_book / len(label_paragraphs) + 1)
        sampled_index_list = np.random.choice(len(label_paragraphs), para_num_per_book, replace=False)
        sampled_paragraphs = [label_paragraphs[index] for index in sampled_index_list]
        dataset.extend(sampled_paragraphs)
        sampled_labels.extend([label_id] * para_num_per_book)

    dataset = dataset[:para_num]
    sampled_labels = sampled_labels[:para_num]

    return dataset, sampled_labels

def train_lda_model(train_corpus, id2word, num_topics):
    return gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=num_topics,
                                            random_state=100, update_every=1, chunksize=1000, passes=10,
                                            alpha='auto', per_word_topics=True, dtype=np.float64)

def get_document_topics(lda_model, corpus):
    document_topics = []
    for item in corpus:
        tmp = lda_model.get_document_topics(item)
        init = np.zeros(lda_model.num_topics)
        for index, v in tmp:
            init[index] = v
        document_topics.append(init)
    return document_topics

def evaluate_model(classifier, train_cla, test_cla, labels_train, labels_test):
    classifier.fit(train_cla, labels_train)
    accuracy = np.mean(cross_val_score(classifier, train_cla, labels_train, cv=10))
    test_accuracy = accuracy_score(labels_test, classifier.predict(test_cla))
    # accuracy = np.mean(cross_val_score(classifier, train_cla + test_cla, labels_train + labels_test, cv=10))
    print(f'Accuracy: {accuracy:.2f}')
    print(f'test_Accuracy: {test_accuracy:.2f}')

def main(token_num, topic_num, stop_words_list, book_para_jieba, book_para_char):
    # 设置段落数量
    para_num = 1000

    dataset, labels = extract_jieba_paragraphs(para_num, token_num, book_para_jieba, book_para_char, stop_words_list, 'word')
    id2word = corpora.Dictionary(dataset)


    dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1, random_state=42)
    train_corpus = [id2word.doc2bow(text) for text in dataset_train]
    test_corpus = [id2word.doc2bow(text) for text in dataset_test]

    lda_model = train_lda_model(train_corpus, id2word, topic_num)

    train_cla = get_document_topics(lda_model, train_corpus)
    test_cla = get_document_topics(lda_model, test_corpus)

    print("word")
    evaluate_model(SVC(), train_cla, test_cla, labels_train, labels_test)

    dataset, labels = extract_char_paragraphs(para_num, token_num, book_para_char, stop_words_list)
    id2word = corpora.Dictionary(dataset)
    dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.1, random_state=42)
    train_corpus = [id2word.doc2bow(text) for text in dataset_train]
    test_corpus = [id2word.doc2bow(text) for text in dataset_test]

    lda_model = train_lda_model(train_corpus, id2word, topic_num)

    train_cla = get_document_topics(lda_model, train_corpus)
    test_cla = get_document_topics(lda_model, test_corpus)

    print("char")
    evaluate_model(SVC(), train_cla, test_cla, labels_train, labels_test)


if __name__ == '__main__':
    # 设置K的初始token为1000
    K = 1000
    # 设置T的初始主题为20
    T = 20
    # 获取书名列表
    with open(r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\1\inf.txt", 'r', encoding='gb18030') as f:
        f = f.read().split(',')
    book_names = f
    book_names_id = {name: idx for idx, name in enumerate(book_names)}
    # 加载停用词
    with open(r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\1\cn_stopwords.txt", 'r', encoding='utf-8') as f:
        stop_words = f.read().split("\n")
    stop_words.append("\u3000")
    stop_words_list = stop_words

    file_path = r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\jyxstxtqj_downcc.com"
    book_para_jieba = {}
    book_para_char = {}
    for file_name in book_names:
        print(file_name)
        with open(os.path.join(file_path, file_name) + ".txt", "r", encoding='gb18030') as file:
            all_text = file.read()
            jieba_paragraphs = []
            char_paragraphs = []
            paragraphs = all_text.split("\n")
            for para in paragraphs:
                if para == '':
                    continue
                jieba_para, char_para = preprocess_text(paragraphs, stop_words_list)
                jieba_paragraphs.append(jieba_para)
                char_paragraphs.append(char_para)
            # 去掉为空字符的段落
            jieba_paragraphs = [sublist for sublist in jieba_paragraphs if sublist]
            char_paragraphs = [sublist for sublist in char_paragraphs if sublist]
            book_para_jieba[file_name] = jieba_paragraphs

            book_para_char[file_name] = char_paragraphs
    main(K, T, stop_words_list, book_para_jieba, book_para_char)