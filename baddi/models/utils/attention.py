import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from baddi.models.utils.commons import get_pooling


class StandardSelfAttention(nn.Module):
    r"""
    Standard Self Attention module to emulate interactions between elements of a sequence

    Arguments
    ----------
        input_size: int
            Size of the input vector at each time step
        output_size: int
            Size of the output at each time step
        outnet: Union[`torch.utils.module`, callable], optional:
            Neural network that will predict the output. If not provided,
            A MLP without activation will be used.
            (Default value = None)
        pooling: str, optional
            Pooling operation to perform. It can be either
            None, meaning no pooling is performed, or one of the supported pooling
            function name (see :func:`ivbase.utils.commons.get_pooling`)
            (Default value = None)

    Attributes
    ----------
        attention_net: 
            linear function to use for computing the attention on input
        output_net: 
            linear function for computing the output values on 
            which attention should be applied
    """

    def __init__(self, input_size, output_size, pooling=None, outnet=None):
        super(StandardSelfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention_net = nn.Linear(self.input_size, self.input_size, bias=False)
        self.output_net = outnet or nn.Linear
        print(outnet)
        self.output_net = self.output_net(self.input_size, self.output_size)
        self.pooling = None
        if pooling:
            self.pooling = get_pooling(pooling)
            # any error here should be propagated immediately

    def forward(self, x, value=None, return_attention=False):
        r"""
        Applies attention on input 

        Arguments
        ----------
            x: torch.FLoatTensor of size B*N*M
                Batch of B sequences of size N each.with M features.Note that M must match the input size vector
            value: torch.FLoatTensor of size B*N*D, optional
                Use provided values, instead of computing them again. This is to address case where the output_net has complex input.
                (Default value = None)
            return_attention: bool, optional
                Whether to return the attention matrix.

        Returns
        -------
            res: torch.FLoatTensor of size B*M' or B*N*M'
                The shape of the resulting output, will depends on the presence of a pooling operator for this layer
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        assert x.size(-1) == self.input_size
        query = x
        key = self.attention_net(x)
        if value is None:
            value = self.output_net(x)
        key = key.transpose(1, 2)
        attention_matrix = torch.bmm(query, key)
        attention_matrix = attention_matrix / math.sqrt(self.input_size)
        attention_matrix = F.softmax(attention_matrix, dim=2)
        applied_attention = torch.bmm(attention_matrix, value)
        if self.pooling is None:
            res = applied_attention
        else:
            res = self.pooling(applied_attention)
        return (res, attention_matrix) if return_attention else res

    def __repr__(self):
        return f'{self.__class__.__name__}(insize={str(self.input_size)}, outsize={str(self.output_size)})'
