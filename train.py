from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from Moudle.Decoder import Decoder
from Moudle.Encoder import Encoder
from Moudle.Attn import AttnDecoderRNN
from dataset.Dataprocess import *
from utils.times import timesince

import argparse

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=75000, help='Number of epochs to train.')
parser.add_argument('--data_path', type=str, default='./data/data/eng-fra.txt', help='train dataset path.')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
parser.add_argument('--teacher_forcing_ratio', type=float, default=.5)
parser.add_argument('--MAX_LENGTH', type=int, default=10)
parser.add_argument('--print_every', type=int, default=5000)
parser.add_argument('--plot_every', type=int, default=5000)
parser.add_argument('--save_path', type=str, default='./save/')
args = parser.parse_args()


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=args.MAX_LENGTH, teacher_forcing_ratio=args.teacher_forcing_ratio):
    """训练函数, 输入参数有8个, 分别代表input_tensor：源语言输入张量，target_tensor：目标语言输入张量，encoder, decoder：编码器和解码器实例化对象
       encoder_optimizer, decoder_optimizer：编码器和解码器优化方法，criterion：损失函数计算方法，max_length：句子的最大长度"""

    # 初始化隐层张量
    encoder_hidden = encoder.InitHidden()

    # 编码器和解码器优化器梯度归0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 初始化编码器输出张量，形状是max_lengthxencoder.hidden_size的0张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # 初始设置损失为0
    loss = 0

    # 循环遍历输入张量索引
    for ei in range(input_length):
        # 根据索引从input_tensor取出对应的单词的张量表示，和初始化隐层张量一同传入encoder对象中
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # 将每次获得的输出encoder_output(三维张量), 使用[0, 0]降两维变成向量依次存入到encoder_outputs
        # 这样encoder_outputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
        encoder_outputs[ei] = encoder_output[0, 0]

    # 初始化解码器的第一个输入，即起始符
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # 初始化解码器的隐层张量即编码器的隐层输出
    decoder_hidden = encoder_hidden

    # 根据随机数与teacher_forcing_ratio对比判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 如果使用teacher_forcing
    if use_teacher_forcing:
        # 循环遍历目标张量索引
        for di in range(target_length):
            # 将decoder_input, decoder_hidden, encoder_outputs即attention中的QKV,
            # 传入解码器对象, 获得decoder_output, decoder_hidden, decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # 因为使用了teacher_forcing, 无论解码器输出的decoder_output是什么, 我们都只
            # 使用‘正确的答案’，即target_tensor[di]来计算损失
            loss += criterion(decoder_output, target_tensor[di])
            # 并强制将下一次的解码器输入设置为‘正确的答案’
            decoder_input = target_tensor[di]

    else:
        # 如果不使用teacher_forcing
        # 仍然遍历目标张量索引
        for di in range(target_length):
            # 将decoder_input, decoder_hidden, encoder_outputs传入解码器对象
            # 获得decoder_output, decoder_hidden, decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # 只不过这里我们将从decoder_output取出答案
            topv, topi = decoder_output.topk(1)
            # 损失计算仍然使用decoder_output和target_tensor[di]
            loss += criterion(decoder_output, target_tensor[di])
            # 最后如果输出值是终止符，则循环停止
            if topi.squeeze().item() == EOS_token:
                break
            # 否则，并对topi降维并分离赋值给decoder_input以便进行下次运算
            # 这里的detach的分离作用使得这个decoder_input与模型构建的张量图无关，相当于全新的外界输入
            decoder_input = topi.squeeze().detach()

    # 误差进行反向传播
    loss.backward()
    # 编码器和解码器进行优化即参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 最后返回平均损失
    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, input_lang, output_lang,Path, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1), "training:"):
        training_pair = tensorsFromPair(random.choice(pairs), input_lang, output_lang)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            # 通过总损失除以间隔得到平均损失
            print_loss_avg = print_loss_total / print_every
            # 将总损失归0
            print_loss_total = 0
            # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
            print('%s (%d %d%%) %.4f' % (timesince(start),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # 当迭代步达到损失绘制间隔时
        if iter % plot_every == 0:
            # 通过总损失除以间隔得到平均损失
            plot_loss_avg = plot_loss_total / plot_every
            # 将平均损失装进plot_losses列表
            plot_losses.append(plot_loss_avg)
            # 总损失归0
            plot_loss_total = 0

    plt.figure(figsize=(12,8),dpi=200)
    plt.plot(plot_losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # 保存到指定路径
    plt.savefig("./s2s_loss.png")
    torch.save(encoder,Path+'encoder_net.pkl')
    torch.save(decoder,Path+'decoder_net.pkl')


if __name__ == '__main__':
    data_path = args.data_path
    hidden_size = args.hidden_size
    epochs = args.epochs
    input_lang, output_lang, pairs = prepareData('eng', 'fra', data_path)
    # 通过input_lang.n_words获取输入词汇总数，与hidden_size一同传入EncoderRNN类中
    # 得到编码器对象encoder1
    encoder1 = Encoder(input_lang.n_words, hidden_size).to(device)
    # 通过output_lang.n_words获取目标词汇总数，与hidden_size和dropout_p一同传入AttnDecoderRNN类中
    # 得到解码器对象attn_decoder1
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    print_every = 5000
    trainIters(encoder1, attn_decoder1, epochs, input_lang=input_lang, output_lang=output_lang,
               print_every=args.print_every, plot_every=args.plot_every,Path=args.save_path)
