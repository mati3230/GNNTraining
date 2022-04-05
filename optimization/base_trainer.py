import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import datetime
from multiprocessing import Process, Value
import json
import os
from .tf_utils import setup_gpu
from .utils import\
    parse_float_value,\
    parse_int_value,\
    parse_bool_value,\
    parse_tuple_int_value,\
    _parse_tuple_int_value,\
    parse_list_int,\
    get_type,\
    save_config,\
    parse_list_tuple


class BaseTrainer():
    """This class executes the training. Moreover, parameters
    of the training can be changed with user commands. The user commands
    should be represented as str list.

    Parameters
    ----------
    args_file : str
        Path (relative or absolute) to a json file where the parameters are
        specified.
    types_file : str
        Path (relative or absolute) to a json file where the types of the
        parameters are specified.

    Attributes
    ----------
    train : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process.
    params : dict
        The parameters of the _args.json file.
    params_types : dict
        The types of the parameters that are specified in the _types.json file.
    train_process : multiprocessing.Process
        Master Process of the training.

    """
    def __init__(self, args_file, types_file, compose_env_args):
        """Constructor.

        Parameters
        ----------
        args_file : str
            Path (relative or absolute) to a json file where the parameters are
            specified.
        types_file : str
            Path (relative or absolute) to a json file where the types of the
            parameters are specified.
        compose_env_args : func
            Function that accepts the params dictionary and outputs the
            arguments to initialize an environment.
        """
        # multiprocessing value for stopping the training across different
        # processes
        self.train = Value("i", True)
        # load the parameters and their types
        with open(args_file) as f:
            self.params = json.load(f)
        with open(types_file) as f:
            self.params_types = json.load(f)
        # check the observation size
        value, msg = parse_list_tuple(
            ["t", self.params["observation_size"]], 0)
        if value == "error":
            raise Exception(msg)
        print("observation value:", value)
        self.params["observation_size"] = value
        self.train_process = None
        try:
            value, msg = parse_list_int(["ignore", self.params["ignore"]], 0)
            if value == "error":
                raise Exception(msg)
            self.params["ignore"] = value
            print("ignore:", self.params["ignore"])
        except:
            pass
        self.env_args = compose_env_args(self.params)


    def start(self, shared_value, params, env_args):
        """Preparation for the training and start of the training.

        Parameters
        ----------
        shared_value : mp.Value
            multiprocessing value for stopping the training across different
            processes
        params : dictionary
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size.

        """
        print("start")
        # gpu memory that should be used by the main process
        mem = params["main_gpu_mem"]
        # force tf to use gpus
        setup_gpu(mem)

        env_name = params["env_name"]
        n_actions = params["n_actions"]
        n_ft_outpt = params["n_ft_outpt"]
        test_interval = params["test_interval"]

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + env_name + "/train/" + current_time

        model_name = params["model_name"]
        model_dir = "./models/" + env_name + "/" + model_name

        policy_type = get_type(params["policy_path"], params["policy_type"])
        env_type = get_type(params["env_path"], params["env_type"])

        # call function to init env_args
        # env_args = self.compose_env_args(params)
        # start the training

        seed = params["seed"]
        tf.random.set_seed(seed)
        np.random.seed(seed)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        tmp_env = env_type(**env_args)
        discrete = tmp_env._discrete
        del tmp_env
        policy, optimization_algo, policy_args = self.get_policy(
            params,
            n_ft_outpt,
            seed,
            n_actions,
            policy_type,
            train_summary_writer,
            discrete)
        if not policy.head_only:
            policy.net.batch_size = params["batch_size"]

        # log some parameters
        save_config(train_log_dir, str(locals()) + ", " + str(params["learning_rate"]))
        master_process = self.get_master_process(
            params,
            shared_value,
            policy,
            policy_type,
            policy_args,
            env_type,
            env_args,
            model_dir,
            model_name,
            train_summary_writer,
            optimization_algo.update,
            seed,
            test_interval)
        print("Master Process ID:", os.getpid())
        master_process.start_loop()

    @abstractmethod
    def get_policy(
            self,
            params,
            n_ft_outpt,
            seed,
            n_actions,
            policy_type,
            train_summary_writer,
            env_type,
            env_args,
            discrete):
        """Abstract method to return the policy that is trained.

        Parameters
        ----------
        params : dict
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size.
        n_ft_outpt : int
            Number of features
        seed : int
            Random seed.
        n_actions : int
            Number of available actions.
        policy_type : type
            Type of the policy.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.

        Returns
        -------
        type
            Instance of the policy that should be optimized.

        """
        pass

    @abstractmethod
    def get_master_process(
            self,
            params,
            shared_value,
            policy,
            policy_type,
            policy_args,
            env_type,
            env_args,
            model_dir,
            model_name,
            train_summary_writer,
            train_f,
            seed,
            test_interval):
        """Abstract method to initialize a master process.

        Parameters
        ----------
        params : dict
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size.
        shared_value : mp.Value
            multiprocessing value for stopping the training across different
            processes
        policy : BasePolicy
            Policy, that will be optimized.
        policy_type : BasePolicy
            Type of the policy.
        policy_args : dict
            Arguments to initialize a policy.
        env_type : BaseEnvironment
            Type of the environment
        env_args : dict
            Arguments to initialize an environment.
        model_dir : str
            Directory where the models are saved.
        model_name : str
            Name of the policy/model.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.
        train_f : func
            Description of parameter `train_f`.
        seed : int
            Random seed.
        test_interval : int
            Interval to conduct tests.

        Returns
        -------
        BaseMasterProcess
            A master process to conduct the optimization on multiple CPUs.

        """
        pass

    def execute_command(self, usr_cmds):
        """Execute a user command that could originate from a, e.g., messaging
        platform such as telegram.

        Parameters
        ----------
        usr_cmds : list
            List that contains user commands as strings. Commands are, for
            example, 'start' or 'stop'. The user can also set arguments
            according to the available types of the algorithm (see
            ppo2_types.json).

        Returns
        -------
        str
            Answer to the command that specifies if the commands could be
            executed.

        """
        usr_cmds = usr_cmds.split()
        for i in range(len(usr_cmds)):
            usr_cmd = usr_cmds[i]
            # start the training
            if usr_cmd == "start":
                self.train.value = True
                self.train_process =\
                    Process(target=self.start, args=(self.train, self.params, self.env_args))
                self.train_process.start()
                return str(self.params.items()) + "\nok"
            # stop the training
            elif usr_cmd == "stop":
                # stop other processes
                if self.train_process:
                    self.train.value = False
                #self.train_process.terminate()
                self.train_process.join()
                return "training stopped"
            # user will change a parameter of the training
            elif usr_cmd in self.params.keys():
                # detect the type of the parameter
                type = self.params_types[usr_cmd]
                if type[-5:] == "float":
                    value, msg = parse_float_value(usr_cmds, i, type)
                elif type[-3:] == "int":
                    value, msg = parse_int_value(usr_cmds, i, type)
                elif type == "str":
                    value = usr_cmds[i + 1]
                    msg = "ok"
                elif type[:5] == "tuple":
                    if type[-3:] == "int":
                        value, msg =\
                            parse_tuple_int_value(usr_cmds, i, type)
                elif type == "list int":
                    value, msg = parse_list_int(usr_cmds, i)
                elif type == "bool":
                    value, msg = parse_bool_value(usr_cmds, i)
                # change the parameter
                if value != "error":
                    self.params[usr_cmd] = value
                i += 1
                return msg
            elif usr_cmd == "help":
                return "start,\nstop\n" + str(self.params.keys()) + ",\nhelp,\nparams"
            elif usr_cmd == "params":
                return str(self.params.items())
        return "unknown command"
