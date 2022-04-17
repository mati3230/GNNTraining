import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, layer_dims, activations, name=None, use_bias=True, dropout=0):
        super(MLP, self).__init__()
        if name is None:
            name = ""
        else:
            name += "_"
        self.layers = []
        self.dropout_layers = []
        if len(layer_dims) < len(activations):
            activations = activations[:len(layer_dims)]
        elif len(layer_dims) > len(activations):
            layers2add = len(layer_dims) - len(activations)
            print("Warning: Will add {0} linear activation functions in layer {1}".format(layers2add, name))
            activations.extend(layers2add*[None])
        if len(layer_dims) > 1:
            for layer_dim, act, i in zip(layer_dims[:-1], activations[:-1], range(len(layer_dims)-1)):
                d = tf.keras.layers.Dense(
                    units=layer_dim,
                    activation=act,
                    name=name + "dense_" + str(i), 
                    use_bias=use_bias)
                self.layers.append(d)
                if(dropout > 0 and dropout < 1):
                    self.dropout_layers.append(tf.keras.layers.Dropout(rate=dropout, name=name+"dropout_" + str(i)))
        d = tf.keras.layers.Dense(
            units=layer_dims[-1],
            activation=activations[-1],
            name=name + "dense_" + str(len(layer_dims)-1),
            use_bias=use_bias)
        self.layers.append(d)
        if(dropout > 0 and dropout < 1):
            self.dropout_layers.append(tf.keras.layers.Dropout(rate=dropout, name=name+"dropout_" + str(len(layer_dims)-1)))
        self.use_dropout = len(self.dropout_layers) > 0

    def build(self, input_shape):
        pass

    def call(self, inputs, is_training=True):
        if self.use_dropout:
            for i in range(len(self.layers)):
                inputs = self.layers[i](inputs)
                inputs = self.dropout_layers[i](inputs, training=is_training)
        else:
            for i in range(len(self.layers)):
                inputs = self.layers[i](inputs)
            
        return inputs


if __name__ == "__main__":
    m = MLP(layer_dims=[10, 5], activations=[tf.nn.relu, tf.nn.relu], name="mlp", dropout=0.1)
    print(len(m.layers))
    x = tf.zeros([1, 100])
    m(x)
    print(m.variables)