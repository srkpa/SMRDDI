import torch.nn as nn
from baddi.models.utils.commons import get_activation


class ClonableModule(nn.Module):
    r"""
    Abstract Module that support a cloning operation.
    Any modules that requires to be cloned should inherit from this abstract class.
    """

    def __init__(self, ):
        super(ClonableModule, self).__init__()
        self.__params = {}

    def clone(self):
        r"""
        Constructs a new module with the same initial parameters.

        Returns
        -------
            new_layer with the same parameters (architecture), but weights reinitialized.
        """
        for key in ['__class__', 'self']:
            if key in self.__params:
                del self.__params[key]

        model = self.__class__(**self.__params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model

    @property
    def output_dim(self):
        r"""
        Output dimension of the module, i.e the number of neurons in its last layer.

        Returns
        -------
            output_dim: int
                Output dimension of this layer

        """
        raise NotImplementedError


class FCLayer(ClonableModule):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.utils.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.utils.Linear)
        out_size: int
            Output dimension of the layer. Should be one supported by :func:`ivbase.utils.commons.get_activation`.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.utils.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer

    """

    def __init__(self, in_size, out_size, activation='ReLU', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size)
        self.activation = get_activation(activation)
        self.init_fn = init_fn
        self.reset_parameters()

    @property
    def output_dim(self):
        r"""
        Dimension of the output feature space in which the input are projected

        Returns
        -------
            output_dim (int): Output dimension of this layer

        """
        return self.out_size

    def reset_parameters(self, init_fn=None):
        r"""
        Initialize weights of the models, in a specific way as defined by the input init function.


        Arguments
        ----------
            init_fn: callable, optional
                Function to initialize the linear weights. If it is not provided
                an attempt to use the object `init_fn` attributes would be first made, before doing nothing.
                (Default value = None)

        .. seealso::
            `torch.utils.init` for more information

        """
        if init_fn := init_fn or self.init_fn:
            init_fn(self.linear.weight)

    def forward(self, x):
        r"""
        Compute the layer transformation on the input.

        Arguments
        ----------
            x: torch.Tensor
                input variable

        Returns
        -------
            out: torch.Tensor
                output of the layer
        """
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            h = self.b_norm(h)
        return h
