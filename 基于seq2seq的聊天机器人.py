import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import jieba



question = [
    "早上好",
    "中午好",
    "下午好",
    "中国首都在哪",
    "美国首都在哪",
    "俄罗斯首都在哪",
    "英国首都在哪",
    "法国首都在哪",
    "韩国首都在哪",
    "印度首都在哪",
    "世界上最长的河流是什么",
    "世界上面积最大的国家",
    "世界上面积最小的国家",
    "国内生产总值最高的国家",
    "铁矿最丰富的国家是",
    "稀土矿最稀有的国家是",
    "你好",
    "您好",
    "西游记的作者是谁",
    "三国演义的作者是谁",
    "红楼梦的作者是谁",
    "水浒传的作者是谁",
    "窗前明月光",
    "举头望明月",
    "我喜欢你",
    "长城是哪个国家的",
    "金字塔是哪个国家的",
    "苹果公司是美国的吗",
    "中国有几个民族",
    "深度学习是什么",
    "英伟达的创始人是谁",
    "微软的创始人是谁",
    "谁发明了造纸术",
    "脸书的创始人是谁",
    "现任美国总统是谁",
    "阿里巴巴的创始人是谁",
    "腾讯的创始人是谁",
    "百度的创始人是谁",
    "字节跳动的创始人是谁",
    "深度求索的创始人是谁",
    "拼多多的创始人是谁",
    "现在几点了",
    "中国的国歌是什么",
    "中国在亚洲吗",
    "中国的货币是",
    "朝鲜的首都在哪",
    "中国有几个合法政党",
    "中国国庆节是什么时候",
    "中国建党节是什么时候",
    "苏联解体是几几年",
    "第一个宇航员是谁",
    "吉利汽车的创始人是谁",
    "你能干什么",
    "现在是几几年",
    "早上好",
    "飞鸟集的作者是谁",
    "狂人日记的作者是谁",
    "百年孤独的作者是谁",
    "基督教的创始人是谁",
    "佛教的创始人是谁",
    "伊斯兰教的创始人是谁",
    "奥丁是什么神话的",
    "宙斯是什么神话的",
    "杭州六小龙在杭州吗",
    "海贼王是日本的吗",
    "七龙珠是日本的吗",
    "火影忍者是日本的吗",
    "深度求索是中国的吗",
    "一拳超人是日本动画吗",
    "深度求索是一家人工智能公司吗",
    "特斯拉是中国企业吗",
    "谷歌是中国企业吗",
    "亚马逊是中国企业吗",
    "索尼是中国企业吗",
    "三星是中国企业吗",
    "华为创始人是谁",
    "马克思是哪国人",
    "钱学森是哪国人",
    "屠呦呦是哪国人",
    "钟南山是哪国人",
    "莫言是哪国人",
    "邓稼先是哪国人",
    "中国图灵奖获得者",
    "机器人会替代人类吗",
    "你好",
    "如何自学深度学习",
    "制造机器人需要学什么",
    "西门子是哪个国家的",
    "奔驰是哪个国家的",
    "大众是哪个国家的",
    "丰田是哪个国家的",
    "本田是哪个国家的",
    "俄罗斯算亚洲国家吗",
    "美国算西方国家吗",
    "米哈游是游戏公司吗",
    "原神是米哈游开发的吗",
    "蔚蓝档案是韩国游戏吗",
    "床前明月光",
    "举头望明月",
    "人类第一个宇航员是",
    "人工智能之父是谁",
    "中国第一个获得图灵奖的人是",
    "李白是哪个朝代的",
    "谢谢你",




]

answer= [
    "早上好",
    "中午好",
    "下午好",
    "北京",
    "华盛顿",
    "莫斯科",
    "伦敦",
    "巴黎",
    "首尔",
    "新德里",
    "尼罗河",
    "俄罗斯",
    "梵蒂冈",
    "美国",
    "巴西",
    "中国",
    "你好",
    "您好",
    "吴承恩",
    "罗贯中",
    "曹雪芹",
    "施耐庵",
    "疑是地上霜",
    "疑是地上霜",
    "我也喜欢你",
    "中国",
    "埃及",
    "是的",
    "五十六个",
    "一种目前很流行的人工智能技术",
    "黄仁勋",
    "比尔盖兹",
    "蔡伦",
    "扎克伯格",
    "特朗普",
    "马云",
    "马化腾",
    "李彦宏",
    "张一鸣",
    "梁文峰",
    "黄铮",
    "不清楚",
    "义勇军进行曲",
    "亚洲",
    "人民币",
    "平壤",
    "九个",
    "十月一号",
    "七月一号",
    "一九九一年",
    "加加林",
    "李书福",
    "和你对话",
    "二零二五年",
    "早上好",
    "泰戈尔",
    "鲁迅",
    "马尔克斯",
    "耶稣",
    "释迦牟尼",
    "穆罕默德",
    "北欧神话",
    "希腊神话",
    "是的",
    "是的",
    "是的",
    "是的",
    "是的",
    "是的",
    "是的",
    "是美国的",
    "美国的",
    "是美国的",
    "日本的",
    "韩国的",
    "任正非",
    "德国人",
    "中国人",
    "中国人",
    "中国人",
    "中国人",
    "中国人",
    "姚期智",
    "不知道",
    "你好",
    "多敲代码",
    "机械电子编程",
    "德国",
    "德国",
    "德国",
    "日本",
    "日本",
    "算",
    "算",
    "是的",
    "是的",
    "是的",
    "疑是地上霜",
    "低头思故乡",
    "加加林",
    "阿兰图灵",
    "姚期智",
    "唐朝",
    "不用谢",



]

# 使用 jieba 分词
def tokenize(sentence):
    """使用 jieba 对句子进行分词，返回一个分词列表"""
    return list(jieba.cut(sentence))

# 构建词汇表
vocab = set()
for q in question:
    vocab.update(tokenize(q))  # 将问题中的词语加入词汇表
for a in answer:
    vocab.update(tokenize(a))  # 将回答中的词语加入词汇表

vocab = list(vocab)  # 转换为列表
vocab_size = len(vocab) + 2  # 添加 <PAD> 和 <EOS> 两个特殊标记

word2idx = {'<PAD>': 0, '<EOS>': 1}  # 定义词到索引的映射
for idx, word in enumerate(vocab):
    word2idx[word] = idx + 2  # 从索引 2 开始分配给实际词汇

idx2word = {v: k for k, v in word2idx.items()}  # 索引到词的反向映射

file_path="word2id3.json"
import json
with open(file_path,"w",encoding="utf-8") as f:
    json.dump(word2idx,f,ensure_ascii=False,indent=4)

file_path1="id2word3.json"
with open(file_path1,"w",encoding="utf-8") as f:
    json.dump(idx2word,f,ensure_ascii=False,indent=4)















# 转换文本为索引序列
def text_to_indices(text, max_len=20):
    """将文本转换为索引序列，并填充或截断到固定长度"""
    tokens = tokenize(text)  # 分词
    indices = [word2idx.get(word, 0) for word in tokens[:max_len]]  # 获取每个词的索引
    indices += [0] * (max_len - len(indices))  # 填充到固定长度
    return indices

def indices_to_text(indices):
    """将索引序列转换回文本"""
    return ''.join([idx2word[idx] for idx in indices if idx != 0 and idx != 1])

# 数据集类
class ChatDataset(Dataset):
    def __init__(self, questions, answers):
        """初始化数据集，将问题和回答转换为索引序列"""
        self.questions = [text_to_indices(q) for q in questions]
        self.answers = [text_to_indices(a) + [1] for a in answers]  # 回答末尾添加 <EOS>

    def __len__(self):
        """返回数据集的长度"""
        return len(self.questions)

    def __getitem__(self, idx):
        """获取指定索引的数据"""
        return torch.tensor(self.questions[idx]), torch.tensor(self.answers[idx])

# Encoder 模型
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """初始化编码器模型"""
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # 词嵌入层
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU 层

    def forward(self, input, hidden):
        """前向传播函数"""
        embedded = self.embedding(input).view(1, 1, -1)  # 获取输入的嵌入表示
        output, hidden = self.gru(embedded, hidden)  # 通过 GRU 层
        return output, hidden

    def init_hidden(self):
        """初始化隐藏状态"""
        return torch.zeros(1, 1, self.hidden_size)

# Decoder 模型
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        """初始化解码器模型"""
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # 词嵌入层
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU 层
        self.out = nn.Linear(hidden_size, output_size)  # 输出层
        self.softmax = nn.LogSoftmax(dim=1)  # Softmax 层

    def forward(self, input, hidden):
        """前向传播函数"""
        output = self.embedding(input).view(1, 1, -1)  # 获取输入的嵌入表示
        output = nn.functional.relu(output)  # 激活函数
        output, hidden = self.gru(output, hidden)  # 通过 GRU 层
        output = self.softmax(self.out(output[0]))  # 输出概率分布
        return output, hidden

    def init_hidden(self):
        """初始化隐藏状态"""
        return torch.zeros(1, 1, self.hidden_size)

# 训练函数
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    """训练单个样本"""
    encoder_hidden = encoder.init_hidden()  # 初始化编码器隐藏状态

    encoder_optimizer.zero_grad()  # 清空编码器梯度
    decoder_optimizer.zero_grad()  # 清空解码器梯度

    input_length = input_tensor.size(0)  # 输入序列长度
    target_length = target_tensor.size(0)  # 目标序列长度

    loss = 0  # 初始化损失

    # 编码阶段
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[word2idx['<PAD>']]])  # 解码器初始输入为 <PAD>
    decoder_hidden = encoder_hidden  # 使用编码器最后的隐藏状态作为解码器的初始隐藏状态

    # 解码阶段
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)  # 获取预测值
        decoder_input = topi.squeeze().detach()  # 使用预测值作为下一次输入

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))  # 计算损失
        if decoder_input.item() == word2idx['<EOS>']:  # 如果遇到 <EOS>，停止解码
            break

    loss.backward()  # 反向传播
    encoder_optimizer.step()  # 更新编码器参数
    decoder_optimizer.step()  # 更新解码器参数

    return loss.item() / target_length  # 返回平均损失

# 训练主循环
def train_iters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    """训练多个样本"""
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)  # 编码器优化器
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)  # 解码器优化器
    training_pairs = [random.choice(dataset) for i in range(n_iters)]  # 随机选择训练样本
    criterion = nn.NLLLoss()  # 损失函数

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter % print_every == 0:
            print(f"Iteration {iter}, Loss: {loss}")  # 打印训练进度和损失

# 保存和加载模型
def save_model(encoder, decoder, path='chatbot_model3.pth'):
    """保存模型参数"""
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, path)

def load_model(path='chatbot_model3.pth', input_size=vocab_size, hidden_size=128, output_size=vocab_size):
    """加载模型参数"""
    checkpoint = torch.load(path)
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return encoder, decoder

# 测试模型
def evaluate(encoder, decoder, sentence, max_length=20):
    """使用模型生成回答"""
    with torch.no_grad():
        input_tensor = torch.tensor(text_to_indices(sentence)).unsqueeze(1)  # 将输入文本转换为张量
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()  # 初始化编码器隐藏状态

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[word2idx['<PAD>']]])  # 解码器初始输入为 <PAD>
        decoder_hidden = encoder_hidden  # 使用编码器最后的隐藏状态作为解码器的初始隐藏状态

        decoded_words = []  # 存储生成的回答

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)  # 获取预测值
            if topi.item() == word2idx['<EOS>']:  # 如果遇到 <EOS>，停止生成
                break
            else:
                decoded_words.append(idx2word[topi.item()])  # 添加预测词
            decoder_input = topi.squeeze().detach()  # 使用预测值作为下一次输入

        return ''.join(decoded_words)  # 返回生成的回答

# 示例运行
if __name__ == "__main__":
    hidden_size = 128  # 隐藏层大小
    encoder = Encoder(vocab_size, hidden_size)  # 初始化编码器
    decoder = Decoder(hidden_size, vocab_size)  # 初始化解码器

    dataset = ChatDataset(question, answer)  # 创建数据集
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # 创建数据加载器

    train_iters(encoder, decoder, n_iters=12000, print_every=1000)  # 训练模型
    save_model(encoder, decoder)  # 保存模型

    encoder, decoder = load_model()  # 加载模型
    # response = evaluate(encoder, decoder, "早上好")  # 测试模型
    # print(f"Input: 早上好\nOutput: {response}")  # 打印结果
    # for i in range(20):
    #     x=input("用户输入：")
    #     response=evaluate(encoder,decoder,x)
    #     print(f"用户输入：{x}, 回答：{response}")
    for i in question:
        response=evaluate(encoder,decoder,i)
        print(f"输入的是：{i},输出的是：{response}")