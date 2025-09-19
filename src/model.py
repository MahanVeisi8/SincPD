import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv1D, Dropout, LeakyReLU,
                                     BatchNormalization, Flatten, MaxPooling1D,
                                     Input, Concatenate, Lambda) # Lambda را اضافه کنید
from sincnet_tensorflow import SincConv1D, LayerNorm

def build_sincnet_model(input_shape=(1000, 8)):
    """
    Builds the SincNet model architecture.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        tf.keras.Model: The uncompiled SincNet model.
    """
    inputs = Input(shape=input_shape)

    # --- تغییر در اینجا اعمال شده است ---
    # از لایه Lambda برای استفاده از tf.split استفاده می‌کنیم
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=8, axis=2))(inputs)
    # ------------------------------------

    list_of_sincs = []
    for i, split_tensor in enumerate(split_tensors):
        # تغییر شکل برای هر سنسور
        # نیازی به expand_dims نیست چون SincConv1D ورودی 2D را می‌پذیرد
        # reshaped_tensor = tf.expand_dims(split_tensor, axis=-1) # این خط دیگر لازم نیست

        x = SincConv1D(N_filt=100, Filt_dim=101, fs=100, stride=5, padding="SAME")(split_tensor)
        x = LayerNorm()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        list_of_sincs.append(x)

    x = Concatenate(axis=-1)(list_of_sincs)

    x = Conv1D(128, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(momentum=0.05)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(256, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(momentum=0.05)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.3))(x)
    x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.3))(x)
    x = BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = LeakyReLU(alpha=0.2)(x)

    prediction = Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=prediction)
    return model