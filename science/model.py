from tensorflow.keras import layers, Model


def cnn(k=6, n_bins=10240):
    """
    n_bins depends on data chunk size and band of interest
    k is number of consecutive data chunks
    Input has shape (k, n_bins, 1)
    Output is scalar anomaly score
    """
    #Input
    inputs = layers.Input(shape=(k, n_bins, 1), name="asd_input")
    x = inputs

    #Layer 1
    x = layers.Conv2D(
        filters=16,              #number of feature detectors, more filters is more computation but more pattern capacity
        kernel_size=(3,5),       #width of each filter across frequency bins (time axis, freq axis)
        strides=(1,2),           #moves the filter n bins at a time (time axis, freq axis)
        padding='same',          #preserves border handling
        activation='relu',
    )(x)

    #Layer 2
    x = layers.Conv2D(filters=24, kernel_size=(3,5), strides=(1,2), padding='same', activation='relu')(x)

    #Layer 3
    x = layers.Conv2D(filters=32, kernel_size=(3,5), strides=(1,2), padding='same', activation='relu')(x)

    #Global Pooling Layer
    x = layers.GlobalAveragePooling2D()(x)     #computes anomaly score for each filter

    #Linear Output Layer
    outputs = layers.Dense(1)(x)         # Dots anomaly tensor with weight tensor and adds scalar bias

    model = Model(inputs, outputs, name="cnn")
    return model
