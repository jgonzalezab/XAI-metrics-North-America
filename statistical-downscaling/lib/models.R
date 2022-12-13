# Load a specific model
load_model <- function(model, input_shape, output_shape) {

    switch (model,

        # DeepESD
    	  DeepESD = {
          inputs <- layer_input(shape = input_shape)
          x = inputs
          l1 = layer_conv_2d(x ,filters = 50, kernel_size = c(3,3), activation = 'relu', padding = 'same')
          l2 = layer_conv_2d(l1,filters = 25, kernel_size = c(3,3), activation = 'relu', padding = 'same')
          l3 = layer_conv_2d(l2,filters = 10, kernel_size = c(3,3), activation = 'relu', padding = 'same')
          l4 = layer_flatten(l3)

          outputs = layer_dense(l4, units = output_shape)
          model <- keras_model(inputs = inputs, outputs = outputs)},

        # CNN-PAN 
        CNNPan = { 
          inputs <- layer_input(shape = input_shape)
          x = inputs

          l1 = layer_conv_2d(x ,filters = 15, kernel_size = c(4, 4), activation = 'relu',
                             padding = 'valid')
          l1_new = layer_conv_2d(l1 ,filters = 20, kernel_size = c(4, 4), activation = 'relu',
                                 padding = 'same')
          l2 = layer_conv_2d(l1_new ,filters = 20, kernel_size = c(4, 4), activation = 'relu',
                             padding = 'valid')

          l3 = layer_conv_2d(l2 ,filters = 20, kernel_size = c(4, 4), activation = 'relu',
                             padding = 'valid')

          l3_new = layer_conv_2d(l3 ,filters = 40, kernel_size = c(4, 4), activation = 'relu',
                                 padding = 'same')

          l4 = layer_flatten(l3_new)

          d1 = layer_dense(l4, units = round(output_shape / 2),
                           activation = 'relu')

          outputs = layer_dense(d1, units = output_shape)
          model <- keras_model(inputs = inputs, outputs = outputs)},

        # CNN-UNET
        CNN_UNET = {
          inputs <- layer_input(shape = input_shape)
          x = inputs
          x = layer_zero_padding_2d(x, padding = list(c(1, 1), c(1, 0)))

          # Encoder
          l1_conv1 = layer_conv_2d(x ,filters = 64, kernel_size = c(2, 2), activation = 'linear',
                                   padding = 'same')
          l1_bn = layer_batch_normalization(l1_conv1)
          l1 = activation_relu(l1_bn)
          l1 = layer_zero_padding_2d(l1, padding = list(c(0, 0), c(5, 5)))
          l1_mp = layer_max_pooling_2d(l1, pool_size = c(2, 2))

          l2_conv1 = layer_conv_2d(l1_mp ,filters = 128, kernel_size = c(2, 2), activation = 'linear',
                                   padding = 'same')
          l2_bn = layer_batch_normalization(l2_conv1)
          l2 = activation_relu(l2_bn)
          l2_mp = layer_max_pooling_2d(l2, pool_size = c(2, 2))

          l3_conv1 = layer_conv_2d(l2_mp ,filters = 256, kernel_size = c(2, 2), activation = 'linear',
                                   padding = 'same')
          l3_bn = layer_batch_normalization(l3_conv1)
          l3 = activation_relu(l3_bn)
          l3_mp = layer_max_pooling_2d(l3, pool_size = c(2, 2))

          l4_conv1 = layer_conv_2d(l3_mp ,filters = 512, kernel_size = c(2, 2), activation = 'linear',
                                   padding = 'same')
          l4_bn = layer_batch_normalization(l4_conv1)
          l4 = activation_relu(l4_bn)
          l4_mp = layer_max_pooling_2d(l4, pool_size = c(2, 2))

          l5_conv1 = layer_conv_2d(l4_mp ,filters = 1024, kernel_size = c(2, 2), activation = 'linear',
                                   padding = 'same')
          l5_bn = layer_batch_normalization(l5_conv1)
          l5 = activation_relu(l5_bn)

          # Decoder
          d1 = layer_conv_2d_transpose(l5, filters = 512, stride = c(2, 2),
                                       kernel_size = c(2, 2), activation = 'relu')
          d1_concat = layer_concatenate(list(d1, l4), axis = 3)
          d1_conv = layer_conv_2d(d1_concat ,filters = 512, kernel_size = c(2, 2), activation = 'relu',
                                  padding = 'same')

          d2 = layer_conv_2d_transpose(d1_conv, filters = 256, stride = c(2, 2),
                                       kernel_size = c(2, 2), activation = 'relu')
          d2_concat = layer_concatenate(list(d2, l3), axis = 3)
          d2_conv = layer_conv_2d(d2_concat ,filters = 256, kernel_size = c(2, 2), activation = 'relu',
                                  padding = 'same')

          d3 = layer_conv_2d_transpose(d2_conv, filters = 128, stride = c(2, 2),
                                       kernel_size = c(2, 2), activation = 'relu')
          d3_concat = layer_concatenate(list(d3, l2), axis = 3)
          d3_conv = layer_conv_2d(d3_concat ,filters = 128, kernel_size = c(2, 2), activation = 'relu',
                                  padding = 'same')

          d4 = layer_conv_2d_transpose(d3_conv, filters = 64, strides = c(2, 2),
                                       kernel_size = c(2, 2), activation = 'relu')
          d4_concat = layer_concatenate(list(d4, l1), axis = 3)
          d4_conv = layer_conv_2d(d4_concat ,filters = 64, kernel_size = c(2, 2), activation = 'relu',
                                  padding = 'same')

          # Final decoding
          final_deconv1 = layer_conv_2d_transpose(d4_conv ,filters = 64, strides = c(2, 2),
                                                  kernel_size = c(2, 2), activation = 'relu')
          final_conv1 = layer_conv_2d(final_deconv1 ,filters = 64, kernel_size = c(2, 2), activation = 'relu',
                                      padding = 'same')

          final_deconv2 = layer_conv_2d_transpose(final_conv1, filters = 64, strides = c(2, 2),
                                                  kernel_size = c(2, 2), activation = 'relu')

          final_conv2 = layer_conv_2d(final_deconv2, filters = 64, kernel_size = c(3, 10), activation = 'relu')
          final_conv3 = layer_conv_2d(final_conv2, filters = 64, kernel_size = c(3, 10), activation = 'relu')
          final_conv4 = layer_conv_2d(final_conv3, filters = 64, kernel_size = c(3, 10), activation = 'relu')
          final_conv5 = layer_conv_2d(final_conv4,filters = 64, kernel_size = c(3, 10), activation = 'relu')
          final_conv6 = layer_conv_2d(final_conv5,filters = 64, kernel_size = c(4, 10), activation = 'relu')

          final_conv7 = layer_conv_2d(final_conv6 ,filters = 1, kernel_size = c(1, 1), activation = 'linear',
                                      padding = 'same')

          model <- keras_model(inputs = inputs, outputs = final_conv7)},

      {print('The selected model does not exist')}
   )

   return(model)

}