import six
import torch.nn as nn
from functools import partial
from torch.nn.functional import normalize
from baddi.utils import is_callable

import torch

SUPPORTED_ACTIVATION_MAP = dir(torch.nn.modules.activation)

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
    'rmsprop': torch.optim.RMSprop,
    'optimizer': torch.optim.Optimizer,
    'lbfgs': torch.optim.LBFGS
}


class Set2Set(torch.nn.Module):
    r"""
    Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Arguments
    ---------
        input_dim: int
            Size of each input sample.
        hidden_dim: int, optional
            the dim of set representation which corresponds to the input dim of the LSTM in Set2Set. 
            This is typically the sum of the input dim and the lstm output dim. If not provided, it will be set to :obj:`input_dim*2`
        steps: int, optional
            Number of iterations :math:`T`. If not provided, the number of nodes will be used.
        num_layers : int, optional
            Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
            (Default, value = 1)
        activation: str, optional
            Activation function to apply after the pooling layer. No activation is used by default.
            (Default value = None)
    """

    def __init__(self, input_dim, hidden_dim=None, steps=None, num_layers=1, activation=None):
        super(Set2Set, self).__init__()
        self.steps = steps
        self.input_dim = input_dim
        self.hidden_dim = input_dim * 2 if hidden_dim is None else hidden_dim
        if self.hidden_dim <= self.input_dim:
            raise ValueError(
                'Set2Set hidden_dim should be larger than input_dim')
        # the hidden is a concatenation of weighted sum of embedding and LSTM output
        self.lstm_output_dim = self.hidden_dim - self.input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_dim, self.input_dim,
                            num_layers=num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self._activation = None
        if activation:
            self._activation = get_activation(activation)

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor 
                Input tensor of size (B, N, D)

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the  set2set pooling operation.
        """

        batch_size = x.shape[0]
        n = self.steps or x.shape[1]

        h = (x.new_zeros((self.num_layers, batch_size, self.lstm_output_dim)),
             x.new_zeros((self.num_layers, batch_size, self.lstm_output_dim)))

        q_star = x.new_zeros(batch_size, 1, self.hidden)

        for _ in range(n):
            # q: batch_size x 1 x input_dim
            q, h = self.lstm(q_star, h)
            # e: batch_size x n x 1
            e = torch.matmul(x, torch.transpose(q, 1, 2))
            a = self.softmax(e)
            r = torch.sum(a * x, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)

        if self._activation:
            return self._activation(q_star, dim=1)
        return torch.squeeze(q_star, dim=1)


class GlobalMaxPool1d(nn.Module):
    r"""
    Global max pooling of a Tensor over one dimension
    See: https://stats.stackexchange.com/q/257321/

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.utils.commons.GlobalAvgPool1d`, :class:`ivbase.utils.commons.GlobalSumPool1d`

    """

    def __init__(self, dim=1):
        super(GlobalMaxPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor
                Input tensor

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the pooling operation.
        """
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool1d(nn.Module):
    r"""
    Global Average pooling of a Tensor over one dimension

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.utils.commons.GlobalAvgPool1d`, :class:`ivbase.utils.commons.GlobalSumPool1d`
    """

    def __init__(self, dim=1):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            For more information, see :func:`ivbase.utils.commons.GlobalMaxPool1d.forward`

        """
        return torch.mean(x, dim=self.dim)


class GlobalSumPool1d(nn.Module):
    r"""
    Global Sum pooling of a Tensor over one dimension

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.utils.commons.GlobalMaxPool1d`, :class:`ivbase.utils.commons.GlobalAvgPool1d`

    """

    def __init__(self, dim=1):
        super(GlobalSumPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            For more information, see :func:`ivbase.utils.commons.GlobalMaxPool1d.forward`
        """
        return torch.sum(x, dim=self.dim)


class UnitNormLayer(nn.Module):
    r"""
    Normalization layer. Performs the following operation: x = x / ||x||

    """

    def __init__(self, dim=1):
        super(UnitNormLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Unit-normalizes input x

        Arguments
        ----------
            x: torch.FloatTensor of size N*M
                Batch of N input vectors of size M.

        Returns
        -------
            x: torch.FloatTensor of size N*M
                Batch of normalized vectors along the dimension 1.
        """
        return normalize(x, dim=self.dim)


class Transpose(nn.Module):
    r"""
    Transpose two dimensions of some tensor.

    Arguments
    ----------
        dim1: int
            First dimension concerned by the transposition
        dim2: int
            Second dimension concerned by the transposition

    Attributes
    ----------
        dim1: int
            First dimension concerned by the transposition
        dim2: int
            Second dimension concerned by the transposition
    """

    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        r"""
        Transposes dimension of input x

        Arguments
        ----------
            x: torch.Tensor
                Tensor to be transpose. x should support dim1 and dim2.

        Returns
        -------
            x: torch.Tensor
                transposed version of input

        """
        return x.transpose(self.dim1, self.dim2)


class Chomp1d(nn.Module):
    r"""
    Chomp or trim a batch of sequences represented as 3D tensors

    Arguments
    ----------
        chomp_size: int
            the length of the sequences after the trimming operation

    Attributes
    ----------
        chomp_size: int
            sequence length after trimming
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        r"""
        Trim input x

        Arguments
        ----------
            x: 3D torch.Tensor
                batch of sequences represented as 3D tensors, the first dim is the batch size,
                the second is the length of the sequences, and the last their embedding size

        Returns
        -------
            x: 3D torch.Tensor
                New tensor which second dimension
                has been modified according to the chomp_size specified
        """
        return x[:, :-self.chomp_size, :].contiguous()


class ResidualBlock(nn.Module):
    r"""Residual Block maker
    Let :math:`f` be a module, the residual block acts as a module g such as :math:`g(x) = \text{ReLU}(x + f(x))`

    Arguments
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.utils.Module, optional
            A down/up sampling module, which is needed
            when the output of the base_module doesn't lie in the same space as its input.
            (Default value = None)
        auto_sample: bool, optional
            Whether to force resampling when the input and output
            dimension of the base_module do not match, and no resampling module was provided.
            By default, the `torch.utils.functional.interpolate` function will be used.
            (Default value = False)
        activation: str or callable
            activation function to use for the residual block
            (Default value = 'relu')
        kwargs: named parameters for the `torch.utils.functional.interpolate` function

    Attributes
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.nn.Module
            the resampling module
        interpolate: bool
            specifies if resampling should be enforced.

    """

    def __init__(self, base_module, resample=None, auto_sample=False, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__()
        self.base_module = base_module
        self.resample = resample
        self.interpolate = False
        self.activation = get_activation(activation)
        if resample is None and auto_sample:
            self.resample = partial(nn.functional.interpolate, **kwargs)
            self.interpolate = True

    def forward(self, x):
        r"""
        Applies residual block on input tensor.
        The output of the base_module will be automatically resampled
        to match the input

        Arguments
        ----------
            x: torch.Tensor
                The input of the residual net

        Returns
        -------
            out: torch.Tensor
                The output of the network
        """
        residual = x
        indim = residual.shape[-1]
        out = self.base_module(x)
        outdim = out.shape[-1]
        if self.resample is not None and not self.interpolate:
            residual = self.resample(x)
        elif self.interpolate and outdim != indim:
            residual = self.resample(x, size=outdim)

        out += residual
        if self.activation:
            out = self.activation(out)
        return out


def get_activation(activation, **kwargs):
    r"""
    Get a pytorch activation layer based on its name. This function acts as a shortcut
    and is case insensitive. Implemented layers should use this function as an activation provider.
    When the input value is callable and not a string, it is assumed that it corresponds to the
    activation function, and thus returned as is.

    Arguments
    ---------
        activation: str or callable
            a python callable or the name of the activation function

    Returns
    -------
        act_cls: torch.utils.Module instance
            module that represents the activation function.
    """
    if is_callable(activation):
        return activation
    assert len(activation) > 0 and isinstance(activation, six.string_types), \
        'Unhandled activation function'
    if activation.lower() == 'none':
        return None

    return vars(torch.nn.modules.activation)[activation](**kwargs)


def get_pooling(pooling, **kwargs):
    r"""
    Get a pooling layer by name. The recommended way to use a pooling layer is through
    this function. When the input is a callable and not a string, it is assumed that it
    corresponds to the pooling layer, and thus returned as is.

    Arguments
    ----------
        pooling: str or callable
            a python callable or the name of the activation function
            Supported pooling functions are 'avg', 'sum' and 'max'
        kwargs:
            Named parameters values that will be passed to the pooling function

    Returns
    -------
        pool_cls: torch.utils.Module instance
            module that represents the activation function.
    """
    if is_callable(pooling):
        return pooling
    # there is a reason for this to not be outside
    POOLING_MAP = {"max": GlobalMaxPool1d, "avg": GlobalAvgPool1d,
                   "sum": GlobalSumPool1d, "mean": GlobalAvgPool1d}
    return POOLING_MAP[pooling.lower()](**kwargs)


def get_optimizer(optimizer):
    r"""
    Get an optimizer by name. cUstom optimizer, need to be subclasses of :class:`torch.optim.Optimizer`.

    Arguments
    ----------
        optimizer: :class:`torch.optim.Optimizer` or str
            A class (not an object) or a valid pytorch Optimizer name

    Returns
    -------
        optm `torch.optim.Optimizer`
            Class that should be initialized to get an optimizer.s
    """
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()]
