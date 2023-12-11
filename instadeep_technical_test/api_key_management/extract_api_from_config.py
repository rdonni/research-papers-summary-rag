import configparser


def get_api_key(config_path="../config.ini"):
    print(config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["open_ai"]["api_key"]
