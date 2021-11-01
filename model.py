import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py
# https://github.com/IamHuijben/Deep-Probabilistic-Subsampling/blob/4cb348a7610619d4b62a6c66b944ee7fc5a10a32/DPS_Huijben2020/CIFAR10_MNIST/myModels.py#L388


class ProbsApproxCatMultiLayer(layers.Layer):


    def __init__(self, mux_in, mux_out):
        super(ProbsApproxCatMultiLayer, self).__init__()
        self.mux_in = mux_in
        self.mux_out = mux_out
        initializer = tf.keras.initializers.RandomNormal() # TODO seed?
        # self.logits = tf.Variable(initial_value=initializer(shape=(1,self.mux_in)), trainable=True)
        # self.temperature = tf.Variable(initial_value=1.0, trainable=True)
        self.logits = self.add_weight(name='TrainableLogits',
                                      shape=(1, self.mux_in),
                                      initializer = initializer,
                                      trainable=True) # TODO
        self.temperature = 1.0#tf.Variable(initial_value=1.0, trainable=True)


    @tf.function
    def sampling(self, BS):
        distribution = tf.random.uniform((BS,1,self.mux_in), 0, 1) + 1e-20
        GN = -tf.math.log(-tf.math.log(distribution)+1e-20)
        perturbedLog = self.logits + GN

        topk = tf.squeeze(tf.nn.top_k(perturbedLog, k=self.mux_out)[1],
                          axis=1)  # [BS,mux_out]
        hardSamples = tf.one_hot(topk,depth=self.mux_in)  # [BS,mux_out,mux_in]

        prob_exp = tf.tile(tf.expand_dims(tf.math.exp(self.logits), 0),
                           (BS,self.mux_out, 1))  # [BS,mux_out,mux_in]
        cumMask = tf.cumsum(hardSamples,axis=-2, 
                            exclusive=True)  # [BS,mux_out,mux_in]

        softSamples = tf.nn.softmax((tf.math.log(tf.math.multiply(prob_exp, 1-cumMask+1e-20))+
                                     tf.tile(GN, (1, self.mux_out, 1)))/self.temperature, axis=-1)
        self.temperature = tf.clip_by_value(self.temperature, 0, 2)  # TODO

        return tf.math.reduce_sum(tf.stop_gradient(hardSamples - softSamples) 
                                  + softSamples, axis=1)


    @tf.function
    def call(self, inputs):
        BS = tf.shape(inputs)[0]
        choice = self.sampling(BS)
        outputs = tf.math.multiply(inputs, tf.reshape(choice, (-1, 1, 1, self.mux_in)))

        return outputs


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    inp_1_conv = layers.Conv2D(n_intermediate_filters,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer="he_normal")(inp_1)
    inp_2_conv = layers.Conv2D(n_intermediate_filters,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="same",
                               kernel_initializer="he_normal")(inp_2)
    f = layers.Activation("relu")(layers.add([inp_1_conv, inp_2_conv]))
    g = layers.Conv2D(filters=1,
                      kernel_size=(1,1),
                      strides=(1,1),
                      padding="same",
                      kernel_initializer="he_normal")(f)
    h = layers.Activation("sigmoid")(g)
    return layers.multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection,
                                      conv_below,
                                      below_filters)
    return layers.concatenate([conv_below, attention_across])


def conv2d_block(inputs, filters,
                 use_batch_norm=True,
                 kernel_size=(3,3),
                 activation="swish",
                 kernel_initializer="he_normal",
                 padding="same"):
    c = layers.Conv2D(filters,
                      kernel_size,
                      padding=padding,
                      activation=activation,
                      kernel_initializer=kernel_initializer)(inputs)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters,
                      kernel_size,
                      padding=padding,
                      activation=activation,
                      kernel_initializer=kernel_initializer)(c)
    if use_batch_norm:
        c = layers.BatchNormalization()(c)
    return c


def get_model(img_size,
              n_sample,
              n_out_channels,
              filters=32,
              num_layers=4,
              use_batch_norm=True):
    inputs = keras.Input(shape=img_size + (21,))
    x = inputs

    # x = ProbsApproxCatMultiLayer(21, n_sample)(x)

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters,
                         use_batch_norm=use_batch_norm)
        down_layers.append(x)
        x = layers.MaxPooling2D(2)(x)
        filters = filters * 2

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)

    for conv in reversed(down_layers):
        filters //= 2
        x = layers.Conv2DTranspose(filters, (2, 2),
                                   strides=(2, 2), padding="same")(x)
        x = attention_concat(conv_below=x, skip_connection=conv)
        # x = layers.concatenate([x, conv], axis=3)
        x = conv2d_block(inputs=x, filters=filters,
                         use_batch_norm=use_batch_norm)

    outputs = layers.Conv2D(n_out_channels, (1, 1), activation="relu")(x) 
    # TODO

    model = keras.Model(inputs, outputs)
    return model
