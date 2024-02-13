from configparser import ConfigParser, ExtendedInterpolation

def read_cfg(path, section):
    """
    Read in a config

    Parameters:
        path : string, relative path to config file
        section : string, section of config to read
    """

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config[section]