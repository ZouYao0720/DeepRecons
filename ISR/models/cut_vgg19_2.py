from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from ISR.utils.logger import get_logger
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda, UpSampling2D

class Cut_VGG19:
    """
    Class object that fetches keras' VGG19 model trained on the imagenet dataset
    and declares <layers_to_extract> as output layers. Used as feature extractor
    for the perceptual loss function.

    Args:
        layers_to_extract: list of layers to be declared as output layers.
        patch_size: integer, defines the size of the input (patch_size x patch_size).

    Attributes:
        loss_model: multi-output vgg architecture with <layers_to_extract> as output layers.
    """
    
    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.input_shape_conv = (patch_size,) * 2 + (1,)
        self.layers_to_extract = layers_to_extract
        self.logger = get_logger(__name__)
        
        if len(self.layers_to_extract) > 0:
            self._cut_vgg()
        else:
            self.logger.error('Invalid VGG instantiation: extracted layer must be > 0')
            raise ValueError('Invalid VGG instantiation: extracted layer must be > 0')
    
    def _cut_vgg(self):
        """
        Loads pre-trained VGG, declares as output the intermediate
        layers selected by self.layers_to_extract.
        """
        
        input_tensor = Input(shape=self.input_shape_conv)
        x = Conv2D(3,(3,3),padding='same')(input_tensor)    # x has a dimension of (IMG_SIZE,IMG_SIZE,3)
        
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        
        x = vgg(x)
        outputs = [x.layers[i].output for i in self.layers_to_extract]
        

        self.model = Model(input_tensor, outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # used in weights naming
