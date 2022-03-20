from torch import nn
import torch
from dataset.Dataprocess import *
device = torch.device('cuda')


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = output.view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def InitHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == '__main__':
    lang1 = "eng"
    lang2 = "fra"

    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    fpairs = filterPairs(pairs)
    input_lang, output_lang, pairs = prepareData('eng', 'fra')
    pair = pairs[0]
    pair_tensor = tensorsFromPair(pair,input_lang,output_lang)
    hidden_size = 25
    input_size = 20
    input = pair_tensor[0][0]
    # 初始化第一个隐层张量，1x1xhidden_size的0张量
    hidden = torch.zeros(1, 1, hidden_size).to(device)
    encoder = Encoder(input_size, hidden_size).to(device)
    encoder_output, hidden = encoder(input, hidden)
    print(encoder_output, encoder_output.shape)
