from tensorflow.keras import layers, models, regularizers
from .layers import SincConv1D

def build_binary_model(input_shape=(8000,8), fs=100.0, sinc_filters=100, sinc_kernel=101, l2=1e-4, dropout=0.2):
    x_in = layers.Input(shape=input_shape)
    x = SincConv1D(sinc_filters, sinc_kernel, fs, name='sinc')(x_in)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x); x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(128,5,padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x); x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(256,5,padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x); x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(x_in, x, name='SincPD_Binary')
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m