"""
Defines architecture for VGG16 with regression head and optional autoencoder branch 

Author: Peter Thomas 
Date: 23 April 2022
"""
import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import (
        LSTM,
        Input,
        Dense, 
        Conv2D, 
        Dropout,
        Flatten,
        ConvLSTM2D,
        UpSampling2D,
        MaxPooling2D,
        TimeDistributed,
        BatchNormalization,
)
from tensorflow.keras.models import Model
from CORONAnet.utils import reset_layer_weights
from CORONAnet.model.activations import LeakyReLU


def VGG16(input_shape):

    # first input model
    visible = Input(shape=input_shape, name='input')

    # 1st-block
    conv1_1 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

    #the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)

    #the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)

    #the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)

    #the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'conv5_4')(conv5_3)
    conv5_4 = BatchNormalization()(conv5_4)
    pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

    # if we want to add decoder branch, add here
    vgg16_outputs = list()
    vgg16_outputs.append(drop5_1)

    # Create time distributed model
    vgg16 = Model(inputs=visible, outputs=vgg16_outputs)

    return vgg16


def Decoder(inputs):
    """
    Add autoencoder output layer to model 
    """
    # the 1st decoder block (numbering starts backwards to denote encoder-decoder relationship)
    decoder_outputs = []
    for input_hook in inputs:
        decoder_conv1_1 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv1_1')(input_hook) 
        decoder_conv1_1 = BatchNormalization()(decoder_conv1_1)

        decoder_conv1_2 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv1_2')(decoder_conv1_1)
        decoder_conv1_3 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv1_3')(decoder_conv1_2)
        decoder_conv1_3 = BatchNormalization()(decoder_conv1_3)
        decoder_conv1_4 = Conv2D(512, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv1_4')(decoder_conv1_3)
        decoder_conv1_4 = BatchNormalization()(decoder_conv1_4)
        upsample1_1 = UpSampling2D((2,2), name='upsample1_1')(decoder_conv1_4)
        drop1_1 = Dropout(0.3, name="decoder_drop1_1")(upsample1_1)

        decoder_conv2_1 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name='decoder_conv2_1')(drop1_1)
        decoder_conv2_1 = BatchNormalization()(decoder_conv2_1)
        decoder_conv2_2 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'decoder_decoder_conv2_2')(decoder_conv2_1)
        decoder_conv2_2 = BatchNormalization()(decoder_conv2_2)
        decoder_conv2_3 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'decoder_decoder_conv2_3')(decoder_conv2_2)
        decoder_conv2_3 = BatchNormalization()(decoder_conv2_3)
        decoder_conv2_4 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', name = 'decoder_decoder_conv2_4')(decoder_conv2_3)
        decoder_conv2_4 = BatchNormalization()(decoder_conv2_4)
        upsample2_1 = UpSampling2D((2,2), name='upsample2_1')(decoder_conv2_4)
        decoder_drop2_1 = Dropout(0.3, name='decoder_drop2_1')(upsample2_1)

        #the 2-nd block
        decoder_conv3_1 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv3_1')(decoder_drop2_1)
        decoder_conv3_1 = BatchNormalization()(decoder_conv3_1)
        decoder_conv3_2 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                name='decoder_conv3_2')(decoder_conv3_1)
        decoder_conv3_2 = BatchNormalization()(decoder_conv3_2)
        decoder_conv3_3 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv3_3')(decoder_conv3_2)
        decoder_conv3_3 = BatchNormalization()(decoder_conv3_3)
        decoder_conv3_4 = Conv2D(256, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv3_4')(decoder_conv3_3)
        decoder_conv3_4 = BatchNormalization()(decoder_conv3_4)
        upsample3_1 = UpSampling2D((2,2), name='upsample3_1')(decoder_conv3_4)
        decoder_drop3_1 = Dropout(0.3, name='decoder_drop3_1')(upsample3_1)

        # 1st decoder block
        decoder_conv4_1 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv4_1')(decoder_drop3_1)
        decoder_conv4_1 = BatchNormalization()(decoder_conv4_1)
        decoder_conv4_2 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv4_2')(decoder_conv4_1)
        decoder_conv4_2 = BatchNormalization()(decoder_conv4_2)
        decoder_conv4_3 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv4_3')(decoder_conv4_2)
        decoder_conv4_3 = BatchNormalization()(decoder_conv4_3)
        decoder_conv4_4 = Conv2D(128, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same', 
                                 name='decoder_conv4_4')(decoder_conv4_3)
        conv4_4 = BatchNormalization()(decoder_conv4_4)
        upsample4_1 = UpSampling2D((2,2), name='upsample4_2')(decoder_conv4_4)
        drop4_1 = Dropout(0.3, name='decoder_drop4_1')(upsample4_1)

        decoder_conv5_1 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv5_1')(drop4_1)
        decoder_conv5_1 = BatchNormalization()(decoder_conv5_1)
        decoder_conv5_2 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv5_2')(decoder_conv5_1)
        decoder_conv5_2 = BatchNormalization()(decoder_conv5_2)
        decoder_conv5_3 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                 name='decoder_conv5_3')(decoder_conv5_2)
        decoder_conv5_3 = BatchNormalization()(decoder_conv5_3)
        decoder_conv5_4 = Conv2D(64, kernel_size=3, activation=LeakyReLU(alpha=0.2), padding='same',
                                name='decoder_conv5_4')(decoder_conv5_3)
        decoder_conv5_4 = BatchNormalization()(decoder_conv5_4)
        upsample5_1 = UpSampling2D((2,2), name='upsample5_2')(decoder_conv5_4)
        decoder_drop5_1 = Dropout(0.3, name='decoder_drop5_1')(upsample5_1)

        # final output layer
        decoder_out = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid',
                             name='decoder_conv_out')(decoder_drop5_1)
        decoder_outputs.append(decoder_out)

    return decoder_outputs


def add_recurrent_regression_head(feature_extractor, num_outputs=1, 
                                  input_shape=(None, None, 1), autoencoder_branch=False):
    """
    Add LSTM module and regression head to base feature extractor 
    """
    # Add temporal dimension to input
    temporal_input = Input(
        shape=(None, *input_shape), 
        name="temporal_input"
    )

    # distribute feature extractor across entire sequence
    feature_extractor_outputs = list()
    for i, out in enumerate(feature_extractor.outputs):
        feature_extractor_out = TimeDistributed(
            Model(feature_extractor.input, out),
            name='time_distributed_model_' + str(i)
        )(temporal_input)
        feature_extractor_outputs.append(feature_extractor_out)

    # Add LSTM layer and dropout
    lstm6_1 = ConvLSTM2D(512, kernel_size=3, padding='same',
                        name='lstm6_1')(feature_extractor_outputs[0])
    drop6_1 = Dropout(0.3, name='drop6_1')(lstm6_1)
    
    #Flatten  output
    outputs = list()
    flatten6_1 = Flatten()(drop6_1)
    dense6_1 = Dense(128, name='dense6_1', activation=LeakyReLU(alpha=0.2))(flatten6_1)
    out = Dense(num_outputs, name='regression_output')(dense6_1)
    outputs.append(out)

    # if we are using an autoencoder branch, add the decoder outputs to the 
    # amalgamated list of model outputs
    if len(feature_extractor_outputs) > 1:
        outputs.extend(feature_extractor_outputs[1:])

    # create model 
    model = Model(inputs=temporal_input, outputs=outputs)

    return model


def freeze_feature_extractor(model, freeze_lstm=True):
    """
    Freeze weights of feature extractor portion of model 
    """
    # Freeze feature extractor 
    model.get_layer('time_distributed_feature_extractor').trainable = False
    
    # if we want to freeze the LSTM layer, do so now
    if freeze_lstm:
        model.get_layer('lstm6_1').trainable = False

    return model


def reset_regression_head_weights(model):
    """
    Reset weights of regression head 
    """
    reset_layer_weights(model.get_layer('dense6_1'))
    reset_layer_weights(model.get_layer('regression_output'))


def fetch_model(image_shape, model_descriptor, num_targets=1):
    """
    helper function that fetches appropriate model constructor
    based on the descriptor string passed in 

    :param model_descriptor: (string) descriptor for which model 
    to import from library
    """
    if model_descriptor == "VGG16":
        vgg16 = VGG16(image_shape)
        return add_recurrent_regression_head(vgg16, num_outputs=num_targets, 
                                             input_shape=image_shape)
    elif model_descriptor == "VGG16+AE":
        vgg16 = VGG16(image_shape)

        input_layer = vgg16.inputs
        output_layer = vgg16.outputs

        decoder_outputs = Decoder(output_layer)

        vgg16_AE = Model(inputs=input_layer, outputs=output_layer + decoder_outputs)

        return add_recurrent_regression_head(vgg16_AE, num_outputs=num_targets,
                                             input_shape=image_shape, autoencoder_branch=True)
    else:
        raise ValueError("Provided model descriptor '{}'".format(model_descriptor)
        + " does not match any model in current library")
