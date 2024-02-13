from configparser import ConfigParser, ExtendedInterpolation

def read_cfg(path, section):
    """
    Read a config file and return a config object

    Parameters:
        path : string, relative path to config file
        section : string, section of config to read
    """

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config[section]