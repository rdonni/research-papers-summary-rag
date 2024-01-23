import configparser
import os

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")


def get_api_key(config_path=config_path):
    print(config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["open_ai"]["api_key"]
