import itertools
import os
from pathlib import Path
from test.python.constants import TESTING_CONFIG_DIR

from omegaconf import OmegaConf

from marius.tools.configuration.marius_config import MariusConfig


def get_config(model_config_path, storage_config_path, train_config_path, eval_config_path):
    model_config = OmegaConf.load(model_config_path)
    storage_config = OmegaConf.load(storage_config_path)
    train_config = OmegaConf.load(train_config_path)
    eval_config = OmegaConf.load(eval_config_path)

    base_config = config_from_sub_configs(model_config, storage_config, train_config, eval_config)

    return base_config


def set_dataset_config(base_config, dataset_dir):
    dataset_config_path = dataset_dir / Path("dataset.yaml")
    dataset_config = OmegaConf.load(dataset_config_path)

    # the below attributes need not be manually set as they will be automatically retrieved from dataset_config_path
    dataset_config.num_edges = -1
    dataset_config.num_nodes = -1
    dataset_config.num_relations = -1
    dataset_config.num_train = -1
    dataset_config.num_valid = -1
    dataset_config.num_test = -1
    dataset_config.initialized = False

    base_config.storage.dataset = dataset_config


def config_from_sub_configs(model_config, storage_config, train_config, eval_config):
    base_config = MariusConfig()

    base_config.model = model_config
    base_config.storage = storage_config
    base_config.training = train_config
    base_config.evaluation = eval_config

    return base_config


def get_cartesian_product_of_configs(config_directory, model_names, storage_names, training_names, evaluation_names):
    model_paths = []
    storage_paths = []
    train_paths = []
    evaluation_paths = []

    for filename in os.listdir(config_directory / Path("model")):
        if len(model_names) > 0:
            for name in model_names:
                if name == filename.split(".")[0]:
                    model_paths.append(config_directory / Path("model") / Path(filename))
        else:
            model_paths.append(config_directory / Path("model") / Path(filename))

    for filename in os.listdir(config_directory / Path("storage")):
        if len(storage_names) > 0:
            for name in storage_names:
                if name == filename.split(".")[0]:
                    storage_paths.append(config_directory / Path("storage") / Path(filename))
        else:
            storage_paths.append(config_directory / Path("storage") / Path(filename))

    for filename in os.listdir(config_directory / Path("training")):
        if len(training_names) > 0:
            for name in training_names:
                if name == filename.split(".")[0]:
                    train_paths.append(config_directory / Path("training") / Path(filename))
        else:
            train_paths.append(config_directory / Path("training") / Path(filename))

    for filename in os.listdir(config_directory / Path("evaluation")):
        if len(evaluation_names) > 0:
            for name in evaluation_names:
                if name == filename.split(".")[0]:
                    evaluation_paths.append(config_directory / Path("evaluation") / Path(filename))
        else:
            evaluation_paths.append(config_directory / Path("evaluation") / Path(filename))

    config_paths = itertools.product(model_paths, storage_paths, train_paths, evaluation_paths)

    configs = []
    config_names = []
    for config_path in config_paths:
        config_name = "M-{}-S-{}-T-{}-E-{}.yaml".format(
            config_path[0].__str__().split("/")[-1].split(".")[0],
            config_path[1].__str__().split("/")[-1].split(".")[0],
            config_path[2].__str__().split("/")[-1].split(".")[0],
            config_path[3].__str__().split("/")[-1].split(".")[0],
        )
        print(config_name)

        configs.append(get_config(config_path[0], config_path[1], config_path[2], config_path[3]))
        config_names.append(config_name)

    return configs, config_names


def get_all_configs_for_dataset(
    dataset_dir, model_names=[], storage_names=[], training_names=[], evaluation_names=[], task="lp"
):
    assert (task == "lp") or (task == "nc")

    config_directory = Path(TESTING_CONFIG_DIR) / Path(task)
    configs, config_names = get_cartesian_product_of_configs(
        config_directory, model_names, storage_names, training_names, evaluation_names
    )

    for config in configs:
        set_dataset_config(config, dataset_dir)

    return configs, config_names


def generate_configs_for_dataset(
    dataset_dir, model_names=[], storage_names=[], training_names=[], evaluation_names=[], task="lp"
):
    configs, config_names = get_all_configs_for_dataset(
        dataset_dir, model_names, storage_names, training_names, evaluation_names, task
    )

    for i, config in enumerate(configs):
        OmegaConf.save(config, Path(dataset_dir) / Path(config_names[i]))
