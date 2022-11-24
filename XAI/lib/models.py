import keras
from keras import layers

# Load a specific model
def load_model(model, input_shape, output_shape):

    # DeepESD
    if model == 'DeepESD':

        inputs = keras.Input(shape = input_shape)

        x = layers.Conv2D(50, kernel_size = (3, 3), activation = 'relu',
                          padding = 'same')(inputs)
        x = layers.Conv2D(25, kernel_size = (3, 3), activation = 'relu',
                          padding = 'same')(x)
        x = layers.Conv2D(10, kernel_size = (3, 3), activation = 'relu',
                          padding = 'same')(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(units = output_shape)(x)

        model = keras.Model(inputs = inputs, outputs = outputs)

    # CNN-PAN
    if model == 'CNNPan':

        inputs = keras.Input(shape = input_shape)

        x = layers.Conv2D(15, kernel_size = (4, 4), activation = 'relu',
                          padding = 'valid')(inputs)
        x = layers.Conv2D(20, kernel_size = (4, 4), activation = 'relu',
                          padding = 'same')(x)
        x = layers.Conv2D(20, kernel_size = (4, 4), activation = 'relu',
                          padding = 'valid')(x)

        x = layers.Conv2D(20, kernel_size = (4, 4), activation = 'relu',
                          padding = 'valid')(x)
        x = layers.Conv2D(40, kernel_size = (4, 4), activation = 'relu',
                          padding = 'same')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units = round(output_shape / 2))(x)

        outputs = layers.Dense(units = output_shape)(x)

        model = keras.Model(inputs = inputs, outputs = outputs)

    # CNN-UNET
    if model == 'CNN_UNET':

        inputs = keras.Input(shape = input_shape)
        x = layers.ZeroPadding2D([(1, 1), (1, 0)])(inputs)

        # Encoder
        l1_conv1 = layers.Conv2D(64, kernel_size = (2, 2), activation = 'linear',
                           padding = 'same')(x)
        l1_bn = layers.BatchNormalization()(l1_conv1)
        l1 = layers.ReLU()(l1_bn)
        l1 = layers.ZeroPadding2D(((0, 0), (5, 5)))(l1)
        l1_mp = layers.MaxPooling2D(pool_size = (2, 2))(l1)

        l2_conv1 = layers.Conv2D(128, kernel_size = (2, 2), activation = 'linear',
                           padding = 'same')(l1_mp)
        l2_bn = layers.BatchNormalization()(l2_conv1)
        l2 = layers.ReLU()(l2_bn)
        l2_mp = layers.MaxPooling2D(pool_size = (2, 2))(l2)

        l3_conv1 = layers.Conv2D(256, kernel_size = (2, 2), activation = 'linear',
                           padding = 'same')(l2_mp)
        l3_bn = layers.BatchNormalization()(l3_conv1)
        l3 = layers.ReLU()(l3_bn)
        l3_mp = layers.MaxPooling2D(pool_size = (2, 2))(l3)

        l4_conv1 = layers.Conv2D(512, kernel_size = (2, 2), activation = 'linear',
                           padding = 'same')(l3_mp)
        l4_bn = layers.BatchNormalization()(l4_conv1)
        l4 = layers.ReLU()(l4_bn)
        l4_mp = layers.MaxPooling2D(pool_size = (2, 2))(l4)

        l5_conv1 = layers.Conv2D(1024, kernel_size = (2, 2), activation = 'linear',
                           padding = 'same')(l4_mp)
        l5_bn = layers.BatchNormalization()(l5_conv1)
        l5 = layers.ReLU()(l5_bn)

        # Decoder
        d1 = layers.Conv2DTranspose(filters = 512, strides = (2, 2),
                                    kernel_size = (2, 2), activation = 'relu')(l5)
        d1_concat = layers.Concatenate(axis = 3)([d1, l4])
        d1_conv = layers.Conv2D(filters = 512, kernel_size = (2, 2), activation = 'relu',
                                padding = 'same')(d1_concat)

        d2 = layers.Conv2DTranspose(filters = 256, strides = (2, 2),
                                    kernel_size = (2, 2), activation = 'relu')(d1_conv)
        d2_concat = layers.Concatenate(axis = 3)([d2, l3])
        d2_conv = layers.Conv2D(filters = 256, kernel_size = (2, 2), activation = 'relu',
                                padding = 'same')(d2_concat)

        d3 = layers.Conv2DTranspose(filters = 128, strides = (2, 2),
                                    kernel_size = (2, 2), activation = 'relu')(d2_conv)
        d3_concat = layers.Concatenate(axis = 3)([d3, l2])
        d3_conv = layers.Conv2D(filters = 128, kernel_size = (2, 2), activation = 'relu',
                                padding = 'same')(d3_concat)

        d4 = layers.Conv2DTranspose(filters = 64, strides = (2, 2),
                                    kernel_size = (2, 2), activation = 'relu')(d3_conv)
        d4_concat = layers.Concatenate(axis = 3)([d4, l1])
        d4_conv = layers.Conv2D(filters = 64, kernel_size = (2, 2), activation = 'relu',
                                padding = 'same')(d4_concat)

        # Final decoding
        final_deconv1 = layers.Conv2DTranspose(filters = 64, strides = (2, 2),
                                               kernel_size = (2, 2), activation = 'relu')(d4_conv)
        final_conv1 = layers.Conv2D(filters = 64, kernel_size = (2, 2), activation = 'relu',
                                    padding = 'same')(final_deconv1)

        final_deconv2 = layers.Conv2DTranspose(filters = 64, strides = (2, 2),
                                               kernel_size = (2, 2), activation = 'relu')(final_conv1)

        final_conv2 = layers.Conv2D(filters = 64, kernel_size = (3, 10), activation = 'relu')(final_deconv2)
        final_conv3 = layers.Conv2D(filters = 64, kernel_size = (3, 10), activation = 'relu')(final_conv2)
        final_conv4 = layers.Conv2D(filters = 64, kernel_size = (3, 10), activation = 'relu')(final_conv3)
        final_conv5 = layers.Conv2D(filters = 64, kernel_size = (3, 10), activation = 'relu')(final_conv4)
        final_conv6 = layers.Conv2D(filters = 64, kernel_size = (4, 10), activation = 'relu')(final_conv5)

        final_conv7 = layers.Conv2D(filters = 1, kernel_size = (1, 1), activation = 'linear',
                                    padding = 'same')(final_conv6)

        outputs = layers.Flatten()(final_conv7)

        model = keras.Model(inputs = inputs, outputs = outputs)

    return model