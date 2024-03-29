import tensorflow as tf
import numpy as np
from .mlp import MLP

from optimization.base_policy import BasePolicy
from optimization.tf_utils import fast_dot, graph_convolution4


class GraphNetLight(BasePolicy):
    def __init__(
            self,
            name,
            n_ft_outpt,
            n_actions,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full",
            stateful=False,
            discrete=True,
            head_only=False,
            observation_size=None):
        super().__init__(
            name=name,
            n_ft_outpt=n_ft_outpt,
            n_actions=n_actions,
            seed=seed,
            trainable=trainable,
            check_numerics=check_numerics,
            initializer=initializer,
            stateful=stateful,
            mode=mode,
            discrete=discrete,
            head_only=head_only,
            observation_size=observation_size)

    def action(self, obs, training=False, decision_boundary=0.5, edge_idxs=None):
        #print(obs.nodes.shape)
        out_g1 = graph_convolution4(
            model_fn_node=self.model_fn_node_1,
            #model_fn_neigh=self.model_fn_node_1,
            activation=tf.nn.relu,
            input_graphs=obs,
            training=training,
            att_model_fn=None)
        
        #"""
        out_g2 = graph_convolution4(
            model_fn_node=self.model_fn_node_2,
            #model_fn_neigh=self.model_fn_node_2,
            activation=tf.nn.relu,
            input_graphs=out_g1,
            training=training,
            att_model_fn=None)
        #"""
        
        out_g2_n = tf.concat([out_g2.nodes, out_g1.nodes], axis=-1)

        out_g2.replace(nodes=out_g2_n)

        #half_e = int(obs.n_edge[0] / 2)
        #s = obs.senders[:half_e]
        #r = obs.receivers[:half_e]

        if edge_idxs is None:
            s = obs.senders
            r = obs.receivers
        else:
            #s = obs.senders[edge_idxs]
            #r = obs.receivers[edge_idxs]
            edge_idxs = edge_idxs.astype(np.int32)
            s = tf.gather(obs.senders, indices=edge_idxs)
            r = tf.gather(obs.receivers, indices=edge_idxs)
        fi = tf.gather(out_g2.nodes, indices=s)
        fj = tf.gather(out_g2.nodes, indices=r)
        f = tf.math.add(fi, fj) / 2
        """ normalization of gnn output
        f_n = tf.norm(f, axis=-1)
        f_n = tf.where(f_n == 0, 1e-6, f_n)
        f_n = tf.expand_dims(f_n, axis=-1)
        f = f / f_n
        """
        d = self.mlp_head(f, is_training=training)
        #d = (self.mlp_head(f, is_training=training) + 1) / 2
        
        action = tf.where(d > decision_boundary, 1, 0)
        pi_action = {
            "action": action,
            "probs": d,
            "dot": 0
            }
        return pi_action

    def init_variables(
            self,
            name,
            n_ft_outpt,
            n_actions,
            trainable=True,
            seed=None,
            initializer="glorot_uniform",
            mode="full"):
        dropout = 0
        self.model_fn_node_1 = MLP(
            layer_dims=[8],
            activations=[None],
            name="mlp_node_1",
            dropout=dropout
            )
        self.model_fn_node_2 = MLP(
            layer_dims=[5],
            activations=[None],
            name="mlp_node_2",
            dropout=dropout
            )
        #"""
        self.mlp_head = MLP(
            layer_dims=[1],
            activations=[tf.nn.sigmoid],
            name="mlp_head",
            dropout=dropout
            )
        #"""
        """
        self.mlp_head = MLP(
            layer_dims=[8, 4, 1],
            activations=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid],
            name="mlp_head",
            dropout=dropout
            )
        """

    def init_net(
            self,
            name,
            n_ft_outpt,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        pass

    def get_vars(self, with_non_trainable=True):
        vars_ = []
        vars_.extend(self.model_fn_node_1.variables)
        vars_.extend(self.model_fn_node_2.variables)
        vars_.extend(self.mlp_head.variables)
        return vars_

    def reset(self):
        pass

    def preprocess(self, obs):
        pass

    def snapshot(self, obs, directory, filename):
        pass