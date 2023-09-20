import torch
from torch import nn
from torch.nn import AvgPool1d, BatchNorm1d, Conv1d, MaxPool1d
from torch.nn import Dropout, Sequential
from torch.nn import LSTM, Embedding
from torch.nn.utils.rnn import pack_padded_sequence

from baddi.models.utils.attention import StandardSelfAttention
from baddi.models.utils.base import ClonableModule
from baddi.models.utils.base import FCLayer
from baddi.models.utils.commons import (
    GlobalAvgPool1d, GlobalMaxPool1d, Transpose, UnitNormLayer, get_activation)



class FCNet(nn.Module):
    def __init__(self, input_size, fc_layer_dims, output_dim, activation='ReLU', last_layer_activation='ReLU',
                 dropout=0., b_norm=False, bias=True, init_fn=None, **kwargs):
        super(FCNet, self).__init__()
        layers = []
        in_size = input_size
        for layer_dim in fc_layer_dims:
            fc = FCLayer(in_size=in_size, out_size=layer_dim, activation=activation, dropout=dropout, b_norm=b_norm,
                         bias=bias, init_fn=init_fn)
            layers.append(fc)
            in_size = layer_dim
        self.net = nn.Sequential(*layers, nn.Linear(in_features=in_size, out_features=output_dim))
        if last_layer_activation is not None:
            self.net.add_module('last_layer_activation', get_activation(last_layer_activation, **kwargs))
        self.__output_dim = output_dim

    def forward(self, inputs):
        return self.net(inputs)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the elements are projected

        Returns
        -------
        output_dim: int
            Dimension of the output feature space
        """
        return self.__output_dim


class CNN1d(ClonableModule):
    r"""
    Extract features from a sequence-like data using Convolutional Neural Network.
    Each time step or position of the sequence must be a discrete value.

    Arguments
    ----------
        vocab_size: int
            Size of the vocabulary, i.e the maximum number of discrete elements possible at each time step.
            Since padding will be used for small sequences, we expect the vocab size to be 1 + size of the alphabet.
            We also expect that 0 won't be use to represent any element of the vocabulary expect the padding.
        embedding_size: int
            The size of each embedding vector
        cnn_sizes: int list
            A list that specifies the size of each convolution layer.
            The size of the list implicitly defines the number of layers of the network
        kernel_size: int or list(int)
            he size of the kernel, i.e the number of time steps include in one convolution operation.
            An integer means the same value will be used for each conv layer. A list allows to specify different sizes for different layers.
            The length of the list should match the length of cnn_sizes.
        pooling_len: int or int list, optional
            The number of time steps aggregated together by the pooling operation.
            An integer means the same pooling length is used for all layers.
            A list allows to specify different length for different layers. The length of the list should match the length of cnn_sizes
            (Default value = 1)
        pooling: str, optional
            One of {'avg', 'max'} (for AveragePooling and MaxPooling).
            It indicates the type of pooling operator to use after convolution.
            (Default value = 'avg')
        dilatation_rate: int or int list, optional
            The dilation factor tells how large are the gaps between elements in
            a feature map on which we apply a convolution filter.  If a integer is provided, the same value is used for all
            convolution layer. If dilation = 1 (no gaps),  every 1st element next to one position is included in the conv op.
            If dilation = 2, we take every 2nd (gaps of size 1), and so on. See https://arxiv.org/pdf/1511.07122.pdf for more info.
            (Default value = 1)
        activation: str or callable, optional
            The activation function. activation layer {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus'}
            The name of the activation function
        normalize_features: bool, optional
            Whether the extracted features should be unit normalized. It is preferable to normalize later in your model.
            (Default value = False)
        b_norm: bool, optional
            Whether to use Batch Normalization after each convolution layer.
            (Default value = False)
        use_self_attention: bool, optional
            Whether to use a self attention mechanism on the last conv layer before the pooling
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, pooling_len=1, pooling='avg',
                 dilatation_rate=1, activation='ReLU', normalize_features=True, b_norm=False,
                 use_self_attention=False, dropout=0.0):
        super(CNN1d, self).__init__()
        self.__params = locals()
        activation_cls = get_activation(activation)
        if not isinstance(pooling_len, (list, int)):
            raise TypeError("pooling_len should be of type int or int list")
        if pooling not in ['avg', 'max']:
            raise ValueError("the pooling type must be either 'max' or 'avg'")
        if len(cnn_sizes) <= 0:
            raise ValueError(
                "There should be at least on convolution layer (cnn_size should be positive.)")

        if isinstance(pooling_len, int):
            pooling_len = [pooling_len] * len(cnn_sizes)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(cnn_sizes)
        if pooling == 'avg':
            pool1d = AvgPool1d
            gpool = GlobalAvgPool1d(dim=1)
        else:
            pool1d = MaxPool1d
            gpool = GlobalMaxPool1d(dim=1)

        # network construction
        embedding = Embedding(vocab_size, embedding_size)
        layers = [Transpose(1, 2)]
        in_channels = [embedding_size] + cnn_sizes[:-1]
        for i, (in_channel, out_channel, ksize, l_pool) in \
                enumerate(zip(in_channels, cnn_sizes, kernel_size, pooling_len)):
            pad = ((dilatation_rate ** i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(in_channel, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate ** i))
            if b_norm:
                layers.append(BatchNorm1d(out_channel))
            layers.append(activation_cls)
            layers.append(Dropout(dropout))
            if l_pool > 1:
                layers.append(pool1d(l_pool))

        if use_self_attention:
            gpool = StandardSelfAttention(
                cnn_sizes[-1], cnn_sizes[-1], pooling)

        layers.append(Transpose(1, 2))
        layers.append(gpool)
        if normalize_features:
            layers.append(UnitNormLayer())

        self.__output_dim = cnn_sizes[-1]
        self.extractor = Sequential(embedding, *layers)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the sequences are projected

        Returns
        -------
        output_dim (int): Dimension of the output feature space

        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x (torch.LongTensor of size N*L): Batch of N sequences of size L each.
                L is actually the length of the longest of the sequence in the bacth and we expected the
                rest of the sequences to be padded with zeros up to that length.
                Each entry of the tensor is supposed to be an integer representing an element of the vocabulary.
                0 is reserved as the padding marker.

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors. D is the dimension of the feature space.
                D is given by output_dim.
        """

        return self.extractor(x)


class Lstm(ClonableModule):
    r"""
    Extracts features from a sequence-like data using LSTM Neural Network.
    Each time step or position of the sequence must be a discrete value.
    The architecture may be composed of multiple LSTM layers but all of them will
    have the same number of hidden units.

    Arguments
    ----------
        vocab_size: int
            Size of the vocabulary, i.e the maximum number of discrete elements
            possible at each time step. Since we will be using some padding for small sequences
            in a batch of sequences, we expect that the vocab size is 1 + the number of element possible.
            We also expect that 0 won't be use to represent any element of the vocabulary expect the padding.
        embedding_size: int
            The size of each embedding vector
        lstm_hidden_size: int
            The size of each lstm layer, i.e the number of neurons contained in cell units and the hidden units
        nb_lstm_layers: int
            The number of lstm layers stacked together.
        bidirectional: bool, optional
            Whether we considered a bidirectional or only uni-directional architecture.
            (Default value = True)
        normalize_features: bool, optional
            Whether the extracted features are unit-normalized.
            Note that it's preferable to normalize later in your model.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        embedding (torch.utils.Embedding): The module that maps the discrete element at each time step into a vector
        extractor (torch.utils.LSTM): the stack of lstm modules that process the sequences and extracts features
    """

    def __init__(self, vocab_size, embedding_size, lstm_hidden_size, nb_lstm_layers,
                 bidirectional=True, normalize_features=False, dropout=0.0):

        super(Lstm, self).__init__()
        self.__params = locals()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.extractor = LSTM(input_size=embedding_size,
                              num_layers=nb_lstm_layers,
                              hidden_size=lstm_hidden_size,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout
                              )
        self._norm_layer = UnitNormLayer() if normalize_features else None
        self.__output_dim = lstm_hidden_size * \
                                nb_lstm_layers * (2 if bidirectional else 1)

    @property
    def output_dim(self):
        r"""
        Dimension of the output feature space in which the sequences are projected

        Returns
        -------
            output_dim: int
                Dimension of the feature space

        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x: torch.LongTensor of size N*L
                Batch of N sequences of size L each.
                L is actually the length of the longest of the sequence in the bacth and we expected the
                rest of the sequences to be padded with zeros up to that length.
                Each entry of the tensor is supposed to be an integer representing an element of the vocabulary.
                0 is reserved as the padding marker.

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors. D is the dimension of the feature space.
                D is given by output_dim.
        """
        x_ = x.view(x.size(0), -1)
        lengths = (x_ > 0).long().sum(1)
        lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        x = x[perm_idx]
        batch_size = x.size(0)
        # h_0 = Variable(torch.zeros((x_train.size(0), self.nb_layers * self.num_directions, self.hidden_size)))
        embedding = self.embedding(x)
        packed_x_train = pack_padded_sequence(
            embedding, lengths.data.cpu().numpy(), batch_first=True)
        _, (hidden, _) = self.extractor(packed_x_train)
        # output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)
        hidden = hidden.transpose(0, 1).contiguous()
        phis = hidden.view(batch_size, -1)

        if self._norm_layer is not None:
            phis = self._norm_layer(phis)

        return phis[rev_perm_idx]


def get_module_fn(network_name):
    if network_name == "feedforward":
        return FCNet
    elif network_name == 'conv1d':
        return CNN1d
    elif network_name == 'lstm':
        return Lstm
    else:
        raise NotImplementedError(
            f"Unknown module {network_name}. Please define the function in network_factory.py and register its use here."
        )
