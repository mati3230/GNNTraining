import tensorflow as tf
import numpy as np
from .mlp import MLP

from optimization.base_policy import BasePolicy
from optimization.tf_utils import fast_dot, graph_convolution3


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
        out_g2 = graph_convolution3(
            model_fn_node=self.model_fn_node_1,
            activation=None,
            input_graphs=obs,
            training=training,
            att_model_fn=None)
        if edge_idxs is None:
            probs = out_g2.nodes
        else:
            probs = tf.gather(out_g2.nodes, indices=edge_idxs)
        
        action = tf.where(probs > decision_boundary, 1, 0)
        pi_action = {
            "action": action,
            "probs": probs,
            "dot": None
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
            layer_dims=[11, 11, 1],
            activations=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid],
            name="mlp_node_1",
            dropout=dropout
            )


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
        return vars_

    def reset(self):
        pass

    def preprocess(self, obs):
        pass

    def snapshot(self, obs, directory, filename):
        pass