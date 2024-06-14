import re
import torch
import jieba
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# 设置torch.cuda.is_available()为False
torch.cuda.is_available = lambda: False

def clean_text(text):
    """清理文本，去除无法识别的字符和特殊字符"""
    text = re.sub('----〖新语丝电子文库\(www.xys.org\)〗', '', text)
    text = re.sub('本书来自www.cr173.com免费txt小说下载站', '', text)
    text = re.sub('更多更新免费电子书请关注www.cr173.com', '', text)
    text = re.sub('\u3000', '', text)
    cleaned_text = text.encode("ascii", "ignore").decode()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def preprocess_chinese_corpus(file_path, stop_words_file=r"C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\1\stopwords.txt"):
    """预处理中文语料库，去除停用词"""
    with open(stop_words_file, "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        words = jieba.lcut(content)
        words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def load_dataset_from_corpus(corpus, tokenizer, model_type="gpt2"):
    """将预处理后的语料库加载为数据集"""
    encoded = tokenizer(corpus, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    labels = encoded["input_ids"].clone()

    dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "labels": labels.tolist()
    })
    return dataset

def finetune_model(model, train_dataset, output_dir, tokenizer, model_type="gpt2"):
    """微调模型"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

# 加载预训练模型和分词器
gpt2_model_path = "user/gpt2-distil-chinese-cluecorpussmall"
gpt2_model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall").to(torch.device('cpu'))
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path, vocab_file='./models/gpt2-chinese-cluecorpussmall/vocab.json')

# 预处理中文语料库
corpus_file = r'C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\jyxstxtqj_downcc.com\鹿鼎记.txt'
chinese_corpus = preprocess_chinese_corpus(corpus_file)

# 加载数据集
train_dataset_gpt2 = load_dataset_from_corpus(chinese_corpus, gpt2_tokenizer, model_type="gpt2")

# 微调模型
finetune_model(gpt2_model, train_dataset_gpt2, "./gpt2-finetuned", gpt2_tokenizer, model_type="gpt2")

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_p=0.95):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = ' '.join(generated_text.split())
    return generated_text

start_text = "扬州城中说书先生说到“长鼻子牛妖”这一节书时"
generated_text_transformer = generate_text(gpt2_model, gpt2_tokenizer, start_text)
print(f"Transformer模型生成结果为: {generated_text_transformer}")
