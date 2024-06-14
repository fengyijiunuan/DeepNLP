from gensim.models import Word2Vec
import jieba
import os

file_path = r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\jyxstxtqj_downcc.com"

# 加载文本数据并进行分词
with open(os.path.join(file_path, "倚天屠龙记") + ".txt", "r", encoding='gb18030') as file:
    text = file.read()
    text = text.replace('----〖新语丝电子文库(www.xys.org)〗', '')
    text = text.replace('本书来自www.cr173.com免费txt小说下载站', '')
    text = text.replace('更多更新免费电子书请关注www.cr173.com', '')
    text = text.replace('\u3000', '')
    text = text.replace(' ', '')
    print(text)

# 对语料库进行分段（按行分割）
    paragraphs = text.split('\n')

# 使用 jieba 对每个段落进行分词
    sentences = [list(jieba.cut(paragraph)) for paragraph in paragraphs if paragraph.strip()]

# 打印前几条分词结果以检查
    for sentence in sentences[:5]:
        print(sentence)

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
word2vec_model.save("word2vec.model")
# 加载模型
word2vec_model = Word2Vec.load("word2vec.model")

# 计算两个词之间的余弦相似度
similarity = word2vec_model.wv.similarity('张无忌', '周芷若')
print(f"张无忌和周芷若的相似度: {similarity}")

# 找出与某个词最相似的词
similar_words = word2vec_model.wv.most_similar('张无忌')
print(f"与张无忌最相似的词: {similar_words}")

similar_words = word2vec_model.wv.most_similar('河南')
print(f"与河南最相似的词: {similar_words}")


similar_words = word2vec_model.wv.most_similar('武林')
print(f"与武功最相似的词: {similar_words}")



# 聚类验证
import numpy as np
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载训练好的Word2Vec模型
word2vec_model = Word2Vec.load("word2vec.model")

# 给定的词列表
words = ['张无忌','赵敏','周芷若','张翠山','殷素素','谢逊',
         '静夜','梨花','雪地','茅草','雪地','清风',
         '武功','兵器','兵刃','武当山','宝刀','功夫']

# 提取词向量
word_vectors = np.array([word2vec_model.wv[word] for word in words])
print(word_vectors)
# 设定聚类数量
num_clusters = 3

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)

# 打印聚类结果
for i in range(num_clusters):
    cluster_words = [words[j] for j in range(len(words)) if kmeans.labels_[j] == i]
    print(f"聚类 {i}: {cluster_words}")


def get_paragraph_vector(paragraph, model):
    words = list(jieba.cut(paragraph))
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 示例段落
paragraph1 = "五年来，我们坚持加强党的全面领导和党中央集中统一领导，全力推进全面建成小康社会进程，完整、准确、全面贯彻新发展理念，着力推动高质量发展，主动构建新发展格局，蹄疾步稳推进改革，扎实推进全过程人民民主，全面推进依法治国，积极发展社会主义先进文化，突出保障和改善民生，集中力量实施脱贫攻坚战，大力推进生态文明建设，坚决维护国家安全，防范化解重大风险，保持社会大局稳定，大力度推进国防和军队现代化建设"
paragraph2 = "同志们！十八大召开至今已经十年了。十年来，我们经历了对党和人民事业具有重大现实意义和深远历史意义的三件大事：一是迎来中国共产党成立一百周年，二是中国特色社会主义进入新时代，三是完成脱贫攻坚、全面建成小康社会的历史任务，实现第一个百年奋斗目标。这是中国共产党和中国人民团结奋斗赢得的历史性胜利，是彪炳中华民族发展史册的历史性胜利，也是对世界具有深远影响的历史性胜利"

# paragraph1 = "张无忌携了谢逊之手，正要并肩走开。谢逊忽道：“且慢！”指着少林僧众中的一名老僧叫道：“成昆！你站出来，当着天下众英雄之前，将诸般前因后果分说明白。群雄吃了一惊，只见这老僧弓腰曲背，形容猥琐，相貌与成昆截然不同"
# paragraph2 = "张无忌见周芷若委顿在地，脸上尽是沮丧失意之情，心下大是不忍，当即上前解开她穴道，扶她起身。周芷若一挥手，推开他手臂，径自跃回峨嵋群弟子之间。只听谢逊朗声说道：今日之事，全自成昆与我二人身上所起，种种恩怨纠缠，须当由我二人了结"
# 计算段落向量
vector1 = get_paragraph_vector(paragraph1, word2vec_model)
vector2 = get_paragraph_vector(paragraph2, word2vec_model)

# 计算段落之间的余弦相似度
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

similarity = cosine_similarity(vector1, vector2)
print(f"段落1和段落2的相似度: {similarity}")
