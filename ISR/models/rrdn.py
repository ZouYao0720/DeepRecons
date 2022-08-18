import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model

from ISR.models.imagemodel import ImageModel

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}


def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """
    
    return RRDN(arch_params, patch_size)


def get_network(weights):
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    return arch_params, c_dim, kernel_size, url, name


class RRDN(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, T, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        T: integer, number or RRDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(
            self, arch_params={}, patch_size=None, beta=0.2, c_dim=3, kernel_size=3, init_val=0.05, weights=''
    ):
        if weights:
            arch_params, c_dim, kernel_size, url, fname = get_network(weights)
        
        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)
    
    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """
        
        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name='F_%d_%d_%d' % (t, d, c),
            )(x)
            F_dc = Activation('relu', name='F_%d_%d_%d_Relu' % (t, d, c))(F_dc)
            x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d_%d' % (t, d, c))
        
        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='LFF_%d_%d' % (t, d),
        )(x)
        return x
    
    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.
        """
        
        # SUGGESTION: MAKE BETA LEARNABLE
        x = input_layer
        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = MultiplyBeta(self.beta)(LFF)
            x = Add(name='LRL_%d_%d' % (t, d))([x, LFF_beta])
        x = MultiplyBeta(self.beta)(x)
        x = Add(name='RRDB_%d_out' % (t))([input_layer, x])
        return x
    
    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling part. """
        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_layer)

        return PixelShuffle(self.scale)(x)
    
    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 1), name='LR_input')
        pre_blocks = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        # DIFFERENCE: in RDN an extra convolution is present here
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        # DIFFERENCE: in RDN a conv with kernel size of 1 after a concat operation is used here
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        # Global Residual Learning
        GRL = Add(name='GRL')([post_blocks, pre_blocks])
        # Upscaling
        PS = self._pixel_shuffle(GRL)
        # Compose SR image
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(PS)
        return Model(inputs=LR_input, outputs=SR)

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        super(PixelShuffle, self).__init__(*args, **kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

class MultiplyBeta(tf.keras.layers.Layer):
    def __init__(self, beta, *args, **kwargs):
        super(MultiplyBeta, self).__init__(*args, **kwargs)
        self.beta = beta

    def call(self, x, **kwargs):
        return x * self.beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta': self.beta,
        })
        return config
