import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import graph_nets as gn
try:
    import tensorflow_probability as tfp
except:
    print("Cannot import tfp")


#@tf.function
def kl_div(x, y):
    """Computes the kullback leibler divergence.

    Parameters
    ----------
    x : tf.Tensor
        Distribution.
    y : tf.Tensor
        Distribution.

    Returns
    -------
    tf.Tensor
        Kullback leibler divergence.

    """
    X = tfp.distributions.Categorical(logits=x)
    Y = tfp.distributions.Categorical(logits=y)
    return tfp.distributions.kl_divergence(X, Y, allow_nan_stats=False)


def kl_div_continous(mu_x, sigma_x, mu_y, sigma_y):
    """Computes the kullback leibler divergence.

    Parameters
    ----------
    mu_x : tf.Tensor
        Mean values of normal distribution.
    sigma_x : tf.Tensor
        Standard deviation of normal distribution.
    mu_y : tf.Tensor
        Mean values of normal distribution.
    sigma_y : tf.Tensor
        Standard deviation of normal distribution.

    Returns
    -------
    tf.Tensor
        Kullback leibler divergence.

    """
    X = tfp.distributions.Normal(loc=mu_x, scale=sigma_x)
    Y = tfp.distributions.Normal(loc=mu_y, scale=sigma_y)
    return tfp.distributions.kl_divergence(X, Y, allow_nan_stats=False)


#@tf.function
def sync(a, b):
    """Sync two networks such that $b \leftarrow a$. The networks a and b
    should have the same parameters.

    Parameters
    ----------
    a : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.
    b : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.

    """
    a_vars = a.get_vars()
    b_vars = b.get_vars()
    for a_var, b_var in zip(a_vars, b_vars):
        b_var.assign(a_var)


# @tf.function
def sync2(a_vars, b):
    """Sync two networks such that $b \leftarrow a$. The networks a and b
    should have the same parameters.

    Parameters
    ----------
    a_vars : list
        List of net parameters.
    b : BaseNet
        Net where parameters can be accessed as list via 'net.get_vars()'
        method.

    """
    b_vars = b.get_vars()
    if len(a_vars) != len(b_vars):
        max_ = min(len(a_vars), len(b_vars))
        offset = len("target_policy/")
        for i in range(max_):
            print(i, "\t", a_vars[i].name[offset:], "\t", b_vars[i].name[offset:])
        raise Exception("different number of weights to sync")
    for a_var, b_var in zip(a_vars, b_vars):
        b_var.assign(a_var)


def dot(a, b):
    """Dot product between two vectors.

    Parameters
    ----------
    a : tf.Tensor
        Feature vector of shape batch size cross n_features.
    b : tf.Tensor
        Feature vector of shape batch size cross n_features.

    Returns
    -------
    tf.Tensor
        Dot product.
    """
    b_shape = tf.shape(b)
    bs = b_shape[0]
    #print(b_shape)
    if len(b_shape) == 3:
        a = tf.squeeze(a, axis=1)
        #print(a)
        b = tf.squeeze(b, axis=1)
    dot = []
    for i in range(bs):
        a_i = a[i]
        b_i = b[i]
        dot_i = a_i * b_i
        # print(dot_i)
        dot.append(dot_i[None, :])
    dot = tf.concat(dot, axis=0)
    #print(dot.shape)
    dot_sum = tf.reduce_sum(dot, axis=-1)
    #print(dot_sum)
    a_n = tf.linalg.norm(a, axis=-1)
    b_n = tf.linalg.norm(b, axis=-1)
    #print(a_n)
    #print(b_n)
    dot = dot_sum / (a_n * b_n + 1e-12)
    #print(dot)
    return dot, a_n, b_n

def fast_dot(a, b):
    dot = a*b
    dot = tf.reduce_sum(dot, axis=-1)
    a_n = tf.linalg.norm(a, axis=-1)
    b_n = tf.linalg.norm(b, axis=-1)
    dot /= ((a_n * b_n) + 1e-6)
    return dot, a_n, b_n


def np_fast_dot(a, b):
    dot = a*b
    dot = np.sum(dot, axis=-1)
    a_n = np.linalg.norm(a, axis=-1)
    b_n = np.linalg.norm(b, axis=-1)
    dot /= ((a_n * b_n) + 1e-6)
    return dot, a_n, b_n


def hinge_loss_similarity(a, b, y):
    """Hinge loss from a dot product of two feature vector a, b.

    Parameters
    ----------
    a : tf.Tensor
        Feature vector of shape batch size cross n_features.
    b : tf.Tensor
        Feature vector of shape batch size cross n_features.
    y : tf.Tensor
        Binary vector of shape batch size

    Returns
    -------
    tf.Tensor
        Hinge loss.
    """
    d = dot(a, b)
    # print(d)
    loss = 1 - y * d
    return loss


def cosine_loss(a, b, y, m):
    """Caculate the cosine loss between two feature vectors.

    Parameters
    ----------
    a : tf.Tensor
        Feature vector of shape batch size cross n_features.
    b : tf.Tensor
        Feature vector of shape batch size cross n_features.
    y : tf.Tensor
        Binary vector of shape batch size
    m : float
        Margin that is used to calculate the loss 

    Returns
    -------
    tf.Tensor
        Cosine loss.
    """
    d = dot(a, b)
    loss = y * (1-d) + (1-y) * tf.math.maximum(0, 2*d+m)
    return loss


def euclidean_distance(a, b):
    return tf.linalg.norm(a-b)


def contrastive_loss(a, b, y, distance_f, m=0.05):
    d = distance_f(a, b)
    pos_loss = tf.math.maximum(0, m - d)
    pos_loss = 0.5 * tf.math.square(pos_loss)
    neg_loss = 0.5 * tf.math.square(d)
    loss = (1-y) * neg_loss + y * pos_loss
    return loss


def setup_gpu(memory=None):
    """Reserve some memory on the gpu.

    Parameters
    ----------
    memory : int
        Size of the memory that should be reserved.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            if memory is None:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            else:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def graph_convolution(model_fn, input_graphs, training, node_factor=0.7):
    # Send the node features to the edges that are being sent by that node. 
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)

    nodes_at_receiver_edges = gn.blocks.broadcast_receiver_nodes_to_edges(input_graphs)
    temporary_graph_recv = input_graphs.replace(edges=nodes_at_receiver_edges)

    # Average the all of the edges received by every node.
    nodes_with_aggregated_edges_s = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_sent)
    # Average the all of the edges sent by every node.
    nodes_with_aggregated_edges_r = gn.blocks.SentEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_recv)
    nodes_with_aggregated_edges = nodes_with_aggregated_edges_s + nodes_with_aggregated_edges_r

    # Interpolation between input and neighbour features
    aggregated_nodes = node_factor * input_graphs.nodes + (1 - node_factor) * nodes_with_aggregated_edges
    updated_nodes = model_fn(aggregated_nodes, is_training=training)

    output_graphs = input_graphs.replace(nodes=updated_nodes)

    return output_graphs


def attention2(input_graphs, model_fn, training):
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    nodes_at_receiver_edges = gn.blocks.broadcast_receiver_nodes_to_edges(input_graphs)
    
    V = model_fn(nodes_at_sender_edges, is_training=training)
    U = model_fn(nodes_at_receiver_edges, is_training=training)
    #print(V.shape)
    #print(U.shape)
    U_a, _, _ = fast_dot(U, V)
    #print(U_a.shape)
    nominator = tf.math.exp(U_a)
    #nominator = U_a
    #print(U_a)
    #print(nominator)

    temporary_graph_sent = input_graphs.replace(edges=nominator)
    denominator = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_sum)(temporary_graph_sent)
    denominator = 1 / (denominator + 1e-6)
    deno_graphs = input_graphs.replace(nodes=denominator)
    deno_edges = gn.blocks.broadcast_receiver_nodes_to_edges(deno_graphs)
    att = nominator * deno_edges
    return att


def attention(input_graphs, model_fn, training):
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    nodes_at_receiver_edges = gn.blocks.broadcast_receiver_nodes_to_edges(input_graphs)
    diff = nodes_at_receiver_edges - nodes_at_sender_edges
    out_diff = model_fn(diff, is_training=training)

    nominator = tf.math.exp(out_diff)
    #print(out_diff)
    #print(nominator)

    temporary_graph_sent = input_graphs.replace(edges=nominator)
    denominator = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_sum)(temporary_graph_sent)
    denominator = 1 / (denominator + 1e-6)
    deno_graphs = input_graphs.replace(nodes=denominator)
    deno_edges = gn.blocks.broadcast_receiver_nodes_to_edges(deno_graphs)
    att = nominator * deno_edges
    return att


def graph_convolution2(model_fn_node, model_fn_neigh, activation, input_graphs, training, att_model_fn=None):
    # Send the node features to the edges that are being sent by that node. 
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    if att_model_fn is not None:
        att = attention2(input_graphs=input_graphs, model_fn=att_model_fn, training=training)
        #print("----")
        #print(att[att < 0])
        #print(att[att > 1])
        nodes_at_sender_edges *= att[:, None]

    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)

    # Average the all of the edges received by every node.
    nodes_with_aggregated_edges = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_sum)(temporary_graph_sent)

    z_neigh = model_fn_neigh(nodes_with_aggregated_edges, is_training=training)
    z_node = model_fn_node(input_graphs.nodes, is_training=training)
    updated_nodes = z_node + z_neigh
    if activation is not None:
        updated_nodes = activation(updated_nodes)


    output_graphs = input_graphs.replace(nodes=updated_nodes)

    return output_graphs


def graph_convolution3(model_fn_node, activation, input_graphs, training, att_model_fn=None):
    # Send the node features to the edges that are being sent by that node. 
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    if att_model_fn is not None:
        att = attention2(input_graphs=input_graphs, model_fn=att_model_fn, training=training)
        nodes_at_sender_edges *= att[:, None]

    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)

    # Average the all of the edges received by every node.
    nodes_with_aggregated_edges = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_sent)

    z = model_fn_node(nodes_with_aggregated_edges, is_training=training)
    if activation is not None:
        z = activation(z)


    output_graphs = input_graphs.replace(nodes=z)

    return output_graphs
