from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input


def load_backbone(model_name='resnet_50'):
    if model_name == 'resnet':
        inputs = Input(shape=(None, None, 3), name='images')
        backbone_layer = ResNet50(weights='imagenet', input_shape=(None,None,3), include_top=False)(inputs)

        return backbone_layer


