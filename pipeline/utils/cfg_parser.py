from configparser import ConfigParser, ExtendedInterpolation


def read_ini(path, section):
    """
    Read a config file and return a config object

    Parameters:
        path : string, relative path to config file
        section : string, section of config to read
    """

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config[section]


def read_yml(path):
    raise ValueError("read_yml not yet implemented")