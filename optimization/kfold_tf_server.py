import numpy as np
import tensorflow as tf
import datetime

from .kfold_network_master import KFoldServer


class KFoldTFServer(KFoldServer):
    def __init__(
            self,
            args_file,
            model,
            global_norm_t,
            learning_rate,
            test_interval,
            k_fold,
            model_dir,
            seed,
            p_data,
            dataset,
            ip="127.0.0.1",
            port=5000,
            buffer_size=4096,
            n_nodes=10,
            recv_timeout=4,
            save_param="Mean_Acc",
            experiment_name="1",
            n_epochs=1,
            set_test_interval=None
            ):
        self.args_file = args_file
        self.model = model
        self.global_norm_t = global_norm_t
        self.optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
        self.save_param = save_param
        self.best_stat_val = 0
        
        super().__init__(
            test_interval=test_interval,
            k_fold=k_fold,
            model_dir=model_dir,
            seed=seed,
            p_data=p_data,
            dataset=dataset,
            ip=ip,
            port=port,
            buffer_size=buffer_size,
            n_nodes=n_nodes,
            recv_timeout=recv_timeout,
            experiment_name=experiment_name,
            n_epochs=n_epochs,
            set_test_interval=set_test_interval
            )

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "./logs/tf/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

    def save_model(self, name=None):
        if name is None:
            self.model.save(directory=self.model_dir, filename="tmp_net")
        else:
            self.model.save(directory=self.model_dir, filename=name)

    def get_model_vars(self):
        vars_ = self.model.get_vars(with_non_trainable=False)
        return vars_

    def preprocess_grads(self, grads):
        global_norm = tf.linalg.global_norm(grads)
        if self.global_norm_t > 0:
            grads, _ = tf.clip_by_global_norm(
                grads,
                self.global_norm_t,
                use_norm=global_norm)
        return grads, {"global_norm": global_norm}

    def write_avg_test_results(self, keys, avg_stats):
        n_stats = len(keys)
        with self.train_summary_writer.as_default():
            for i in range(n_stats):
                key = keys[i]
                stat = avg_stats[i]
                tf.summary.scalar("test/" + key, stat, step=self.test_step)

                if key == self.save_param:
                    if stat > self.best_stat_val:
                        self.best_stat_val = stat
                        self.save_model(name="best_{0:.3f}".format(stat))
        self.train_summary_writer.flush()

    def apply_grads(self, grads, vars_):
        self.optimizer.apply_gradients(zip(grads, vars_))

    def write_loss(self, name, loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar("train/avg_" + name, loss, step=self.train_step)

    def write_adds(self, adds):
        global_norm = adds["global_norm"]
        with self.train_summary_writer.as_default():
            tf.summary.scalar("train/global_norm", global_norm, step=self.train_step)

    def on_train_write_end(self):
        self.train_summary_writer.flush()

    def to_tensor(self, t, dtype):
        tftype = tf.float32
        t = tf.convert_to_tensor(t, dtype=tftype)
        return t

    def get_init_msg(self):
        msg = super().get_init_msg()
        msg += "," + self.args_file
        return msg

    def reset_method(self):
        self.model.load(directory=self.model_dir, filename="init_net")