# config_loader.py
import yaml


def get_data_dir(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['data_dir']
