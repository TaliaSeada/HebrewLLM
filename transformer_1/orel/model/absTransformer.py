from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from model.config import DROPOUT, EPOCHS, BATCH_SIZE


class Transformer(ABC, nn.Module):
    def __init__(self, input_size=1024, output_size=None, num_layers=2, num_heads=1, dim_feedforward=128, dropout=DROPOUT, activation=F.relu):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size else input_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        

    @abstractmethod
    def build_transformer_encoder(self):
        pass

    @abstractmethod
    def forward(self, src):
        pass


class AbstractHiddenStateTransformer(Transformer):
    def __init__(self, input_size=1024, output_size=None, num_layers=2, num_heads=1, dim_feedforward=128, dropout=DROPOUT, activation=F.relu):
        super(AbstractHiddenStateTransformer, self).__init__(input_size, output_size, num_layers, num_heads, dim_feedforward, dropout, activation)

    @abstractmethod
    def build_transformer_encoder(self):
        pass

    @abstractmethod
    def forward(self, src):
        pass