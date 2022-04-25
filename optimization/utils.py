import numpy as np
import os
import tensorflow as  tf
import math
import h5py
from graph_nets import utils_tf
import libgeo

try:
    import cv2
except ImportError:
    print("Cannot import opencv")


def load_graph_batch(i, dir, files):
    file = files[i]
    file_dir = dir + "/" + file

    hf = h5py.File(file_dir, "r")

    node_features = np.array(hf["node_features"], copy=True)
    senders = np.array(hf["senders"], copy=True)
    receivers = np.array(hf["receivers"], copy=True)
    edge_idxs = np.array(hf["edge_idxs"], copy=True)
    y = np.array(hf["unions"], copy=True)

    hf.close()
    input_graphs = {"nodes": node_features, "senders": senders, "receivers": receivers, "edges": None, "globals": None}
    input_graphs = utils_tf.data_dicts_to_graphs_tuple([input_graphs])

    return input_graphs, y, edge_idxs


def split_examples(train_idxs, train_n, client_id, n_clients):
    examples_per_client = math.floor(train_n / n_clients)
    start_i = examples_per_client * client_id
    stop_i = examples_per_client * (client_id + 1)
    if client_id == n_clients - 1:
        stop_i = train_n
    return train_idxs[start_i:stop_i]


def socket_send(file, sock, buffer_size=4096):
    f = open(file,"rb")
    l = f.read(buffer_size)
    while (l):
        sock.send(l)
        l = f.read(buffer_size)
    f.close()


def socket_recv(file, sock, buffer_size=4096, timeout=4, msg_size=None):
    f = open(file, "wb")
    if msg_size is None:
        sock.settimeout(None)
    recv_size = 0
    l = sock.recv(buffer_size)
    recv_size += len(l)
    if msg_size is None:
        sock.settimeout(timeout)
    while (True):
        f.write(l)
        if msg_size is None:
            try:
                l = sock.recv(buffer_size)
                recv_size += len(l)
            except:
                break
        else:
            # msg received?
            if recv_size == msg_size:
                break
            l = sock.recv(buffer_size)
            recv_size += len(l)
    f.close()
    if msg_size is None:
        sock.settimeout(None)
    return recv_size


def concat_transitions(transitions):
    """Concate the transitions  of type 'list(dict)' to a dict(str, list).

    Parameters
    ----------
    transitions : list(dict)

    Returns
    -------
    dict(str, list)
        Dictionary where each value is a batch of transitions.

    """
    if len(transitions) == 1:
        return transitions[0]
    keys = transitions[0].keys()
    result = {}
    for key in keys:
        feature = transitions[0][key]
        #print("concat", key)

        reshape_l = [False]
        if key == "observations":
            cfeature = []
            reshape_l = []
            for j in range(len(feature)):
                nshape = (0, ) + feature[j].shape[1:]
                reshape_l.append(len(nshape) == 1)
                if reshape_l[j]:
                    nshape = (0, 1)
                cfeature.append(np.zeros(nshape, feature[j].dtype))
        else:
            nshape = (0, ) + feature.shape[1:]
            reshape_l[0] = len(nshape) == 1
            if reshape_l[0]:
                nshape = (0, 1)
            #print("feature shape", nshape)
            cfeature = np.zeros(nshape, feature.dtype)
        for i in range(len(transitions)):
            trans = transitions[i]
            feature = trans[key]
            if key == "observations":
                for j in range(len(feature)):
                    if reshape_l[j]:
                        feature[j] = feature[j].reshape((feature[j].shape[0], 1))
                    cfeature[j] = np.vstack((cfeature[j], feature[j]))
            else:
                if reshape_l[0]:
                    feature = feature.reshape((feature.shape[0], 1))
                cfeature = np.vstack((cfeature, feature))

        if key == "observations":
            for j in range(len(cfeature)):
                if reshape_l[j]:
                    cfeature[j] = cfeature[j].reshape((cfeature[j].shape[0], ))
        else:
            if reshape_l[0]:
                cfeature = cfeature.reshape((cfeature.shape[0], ))
        result[key] = cfeature
    return result


def returns_advs(gamma, lmbda, rewards, dones, values, norm_rets, norm_adv):
    """Calculates the returns and advantages of some transitions.

    Parameters
    ----------
    gamma : float
        Discount factor
    lmbda : float
        Factor of the generalized advantage estimation.
    rewards : np.ndarray
        An array with rewards.
    dones : np.ndarray
        An array with flags that indicate if an episode is done.
    values : np.ndarray
        State values of various situations
    norm_rets : boolean
        Should the returns be normalized?
    norm_adv : boolean
        Should the advantages be normalized?

    Returns
    -------
    np.ndarray, np.ndarray
        The returns and advantages of the transitions.

    """
    prev_return = 0
    prev_value = 0
    gae = 0
    returns = []
    advantages = []
    for i in reversed(range(rewards.shape[0])):
        reward = rewards[i]
        done = dones[i]
        value = values[i]

        prev_return, _, gae = ppo2_vars(
            done=done,
            prev_return=prev_return,
            gamma=gamma,
            reward=reward,
            value=value,
            prev_value=prev_value,
            lmbda=lmbda,
            gae=gae)

        returns.insert(0, prev_return)
        prev_value = value
        advantages.insert(0, gae)
    returns = np.array(returns, np.float32)
    if norm_rets:
        returns = normalize(returns)
    advantages = np.array(advantages, np.float32)
    if norm_adv:
        advantages = normalize(advantages)
    return returns, advantages


def ppo2_vars(done, prev_return, gamma, reward, value, prev_value, lmbda, gae):
    """Computes the return and generalized advantage estimation. See Schulman
    et al. for more information.

    Parameters
    ----------
    done : boolean
        Flag that indicates if an episode is done.
    prev_return : float
        Last return value.
    gamma : float
        Discount factor
    reward : float
        Reward of the environment.
    value : float
        State value.
    prev_value : float
        Last state value.
    lmbda : float
        Factor of the generalized advantage estimation.
    gae : float
        Generalized advantage estimation value.

    Returns
    -------
    float, float, float
        The return value, the delta to compute the generalized advantage
        estimation and the generalized advantage estimation itself.

    """
    # flip the done values to use the mask as factor
    mask = 1 - done
    return_val = reward + (gamma * prev_return * mask)
    # calculate the generalized advantage estimation
    delta = reward + (gamma * prev_value * mask) - value
    gae = delta + (gamma * lmbda * mask * gae)
    return return_val, delta, gae


def normalize(x, eps=1e-5, axis=0):
    """Normalize via standardisation.

    Parameters
    ----------
    x : np.ndarray
        Array that should be normalized.
    eps : float
        Constant that is used in case of a 0 standard deviation.
    axis : int
        Normalization direction.

    Returns
    -------
    np.ndarray
        Description of returned object.

    """
    return (x - np.mean(x, axis=axis)) / (np.std(x, axis=axis) + eps)


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def save_config(log_dir, config):
    """Save a custom configuration such as learning rate.

    Parameters
    ----------
    log_dir : str
        Directory where the configuration should be placed.
    config : str
        String with the configuration.
    """
    text_file = open(log_dir + "/config.txt", "w")
    text_file.write(config)
    text_file.close()


def importer(name, root_package=False, relative_globals=None, level=0):
    """Imports a python module.

    Parameters
    ----------
    name : str
        Name of the python module.
    root_package : boolean
        See https://docs.python.org/3/library/functions.html#__import__.
    relative_globals : type
        See https://docs.python.org/3/library/functions.html#__import__.
    level : int
        See https://docs.python.org/3/library/functions.html#__import__.

    Returns
    -------
    type
        Python module. See
        https://docs.python.org/3/library/functions.html#__import__.

    """
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals,
                      fromlist=[] if root_package else [None],
                      level=level)


def get_type(path_str, type_str):
    """Load a specific class type.

    Parameters
    ----------
    path_str : str
        Path to the python file of the desired class.
    type_str : str
        String of the class name.

    Returns
    -------
    type
        Requested class type.

    """
    module = importer(path_str)
    mtype = getattr(module, type_str)
    return mtype


def parse_float_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where float values are stored.
    i : int
        The i-th value should be the name of the float value. The (i+1)-th
        value should be the float value itself.
    type : str
        Should be whether:
            real pos float: > 0
            real neg float: < 0
            pos float: >= 0
            neg float: <= 0

    Returns
    -------
    tuple(float, str)
        Returns the float value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    try:
        value = float(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos float" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg float" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos float" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg float" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def _parse_int_value(value, type):
    """Parse a int value.

    Parameters
    ----------
    value : int
        An int value.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    type
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = int(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos int" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg int" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos int" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg int" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def parse_int_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(int, str)
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_int_value(value, type)


def parse_bool_value(usr_cmds, i):
    """Parse a boolean value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(boolean, str)
        Returns the boolean value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value - expected True or False")
    if value != "True" and value != "False":
        return ("error", "Invalid value - expected True or False")
    if value == "True":
        return (True, "ok")
    return (False, "ok")


def parse_list_int(usr_cmds, i):
    """Parse a list of int values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(list(int), str)
        Returns the list(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return "error", "No value - expected list"
    if value == "":
        return None, "ok"
    if len(value) == 1:
        return "error", "List should be at least '[]'"
    if len(value) == 2:
        return [], "ok"
    if len(value) >= 2:
        if value[0] != "[":
            return "error", "List should begin with ["
        if value[-1] != "]":
            return "error", "List should end with ]"
        l_values = value[1:-1]
        l_values = l_values.split(",")
        result = []
        for i in range(len(l_values)):
            l_val = l_values[i]
            if l_val == "" or l_val == " ":
                continue
            val = 0
            if l_val.isdigit():
                val = int(l_val)
            else:
                return "error", str(l_val) + " cannot be converted to int"
            result.append(val)
        return result, "ok"


def parse_list_tuple(usr_cmds, i):
    """Parse a list of tuple values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(list(tuple), str)
        Returns the list(tuple) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return "error", "No value - expected list of tuples"
    if value == "":
        return None, "ok"
    if len(value) == 1:
        return "error", "List should be at least '[]'"
    if len(value) == 2:
        return [], "ok"
    if len(value) >= 2:
        if value[0] != "[":
            return "error", "List should begin with ["
        if value[-1] != "]":
            return "error", "List should end with ]"
        l_values = value[1:-1]
        if l_values[0] != "(":
            return "error", "Tuple should begin with ("
        if l_values[-1] != ")":
            return "error", "Tuple should end with )"
        #print(l_values)
        start_idx = 0
        result = []
        while True:
            #print(start_idx)
            stop_idx = -1
            si = -1
            for i in range(start_idx, len(l_values)):
                if l_values[i] == "(":
                    si = i
                if l_values[i] == ")":
                    stop_idx = i
                if stop_idx != -1:
                    break
            if stop_idx <= si:
                return "error", "A tuple should at least have the form '(x, )'"
            tuple_str = l_values[si:stop_idx+1]
            t, msg = parse_tuple_int_value(["", tuple_str], 0, "pos int")
            if t == "error":
                return t, msg
            result.append(t)
            start_idx = stop_idx + 1
            if stop_idx == len(l_values) - 1:
                break
        return result, "ok"


def _parse_tuple_int_value(value, type):
    """Parse an int tuple.

    Parameters
    ----------
    value : tuple(int)
        Tuple that should be parsed.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    if len(value) < 4:
        return ("error", "Tuple must be at least '(x, )'")
    if value[0] != "(" or value[-1] != ")":
        return ("error", "Tuple must be at least '(x, )'")
    t_values = value[1:-1]
    t_values = t_values.split(",")
    if len(t_values) < 2:
        return ("error", "Tuple must be at least '(x, )'")
    result = []
    for i in range(len(t_values)):
        t_val = t_values[i]
        if t_val == "" or t_val == " ":
            continue
        int_type = type[6:]
        value, msg = _parse_int_value(t_val, int_type)
        if value == "error":
            return (value, msg)
        result.append(value)
    result = tuple(result)
    return (result, "ok")


def parse_tuple_int_value(usr_cmds, i, type):
    """Parse a tuple with int values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the tuple. The (i+1)-th
        value should be the tuple itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_tuple_int_value(value, type)


#@tf.function
def knn_query(P, p_query, k):
    """Extract the k nearest neighbours of p_query.

    Parameters
    ----------
    P : np.ndarray
        A point cloud.
    p_query : np.ndarray
        Query point.
    k : int
        The number of neighbours that should be found.

    Returns
    -------
    np.array
        The k nearest neighbour of p_query.

    """
    d_sort, _ = distance_sort(P=P, p_query=p_query)
    #print(np.sort(d)[:3])
    k_idxs = d_sort[:k]
    knn = P[k_idxs]
    return knn, k_idxs


def distance_sort(P, p_query):
    d_points = P[:, :3] - p_query[:3]
    d_square = np.square(d_points)
    d = np.sum(d_square, axis=1)
    d_sort = np.argsort(d)
    return d_sort, d


#@tf.function
def knn(P, C, k, f):
    """Extract the k nearest neighbours.

    Parameters
    ----------
    P : np.ndarray
        A point cloud.
    C : np.ndarray
        Color values of the point cloud.
    k : int
        The number of neighbours that should be found.
    f : int
        Sampling value. Fraction between the number of points that should be
        sampled and the size of the point cloud.

    Returns
    -------
    tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        Sampled points and their corresponding colour values. Vector with the
        smallest and the largest values in the point cloud.

    """
    min_v = tf.reduce_min(P, axis=0)
    max_v = tf.reduce_max(P, axis=0)
    mid = (min_v + max_v) / 2
    P_ = tf.subtract(P, mid)
    P_abs = tf.abs(P_)
    distances = tf.reduce_sum(P_abs, axis = 1)
    distances = tf.cond(tf.math.greater(k, tf.shape(P)[0]), lambda: tf.tile(input=distances, multiples=f), lambda: distances)
    distances, idxs = tf.math.top_k(distances, k)
    m = tf.constant([f[0], 1], dtype=tf.int32)
    P_knn = tf.cond(tf.math.greater(k, tf.shape(P)[0]), lambda: tf.tile(input=P, multiples=m), lambda: P)
    C_knn = tf.cond(tf.math.greater(k, tf.shape(C)[0]), lambda: tf.tile(input=C, multiples=m), lambda: C)
    P_knn = tf.gather(P_knn, idxs)
    C_knn = tf.gather(C_knn, idxs)
    return P_knn, C_knn, min_v, max_v


def sample_farthest_points(P, C, idxs, sampling_size):
    """Sample the farthest points from the center of a superpoint.

    Parameters
    ----------
    P : np.ndarry
        The point cloud.
    C : np.ndarray
        Color values of the points.
    idxs : np.ndarray
        Indices of a superpoint.
    sampling_size : int
        number of points that should be sampled.

    Returns
    -------
    np.ndarray, np.ndarray
        Sampled point cloud with the corresponding color values.

    """
    P_ = P[idxs]
    C_ = C[idxs]
    f = sampling_size / P_.shape[0]
    f = np.array([f])
    f = np.ceil(f).astype(np.int32)
    P_, C_, min_v, max_v = knn(P=P_, C=C_, k=sampling_size, f=f)
    P_ = (P_ - min_v) / (max_v - min_v)
    return P_, C_


def render_image(image, scale=1, title="image"):
    """Renders an image with OpenCV.

    Parameters
    ----------
    image : np.ndarry
        Image that should be rendered.
    scale : float
        Scale factor to change the size of the image.
    title : np.ndarray
        Title of the plot.

    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    # print(dim)
    image = cv2.resize(image, dim)
    cv2.imshow(title, image)
    # waits until a key is pressed
    cv2.waitKey(0)
    # destroys the window showing image
    cv2.destroyAllWindows()


def render_images(all_images, scale=1, perspectives=4, n_images=3):
    """Render images of multiple views.

    Parameters
    ----------
    all_images : np.ndarray
        An array of images that contains the views. The shape should be
        (I, P, W, H, C).
        V: number of images
        P: number of perspectives
        W: width of an image
        H: height of an image
        C: channels of an image
    scale : float
        Scale parameter for the rendering of the images.
    perspectives : int
        The number of views/perspectives per image.
    n_images : int
        Number of images.

    """
    for j in range(n_images):
        i = j*perspectives
        img = all_images[i:i+perspectives]
        im1 = np.concatenate((img[0], img[1]), axis=0)
        im2 = np.concatenate((img[2], img[3]), axis=0)
        im = np.concatenate((im1, im2), axis=1)
        render_image(im, scale=scale, title="view: " + str(i))


def visualize_sparse_voxel_grid(mlab, svg_features, svg_indxs, title=None, n_batches=1):
    """Renders aa sparse voxelgrid with mayavi.

    Parameters
    ----------
    mlab : type
        Mayavi plot object.
    svg_features : np.ndarray
        Features of the sparse voxel grid (e.g. mean values of the voxels).
    svg_indxs : np.ndarray
        Indices of the voxels.
    title : str
        Title of the plot.
    n_batches : int
        If batch size is greater than 1. Control how many examples from a batch should
        be visualized.

    """
    for b in range(svg_features.shape[0]):
        for fm in range(svg_features.shape[-1]):
            vf = svg_features[b, :, fm]
            max_vf = np.max(vf)
            min_vf = np.min(vf)
            vf = (vf - min_vf) / (max_vf - min_vf)
            vi = svg_indxs[b]
            xx, yy, zz = vi[:, 0], vi[:, 1], vi[:, 2]
            mlab.points3d(xx, yy, zz, vf, mode="cube", scale_factor=1)
            if title is not None:
                mlab.title(title + " b" + str(b) + " fm" + str(fm))
            else:
                mlab.title("b" + str(b) + " fm" + str(fm))
            mlab.show()
        if b >= n_batches:
            break
