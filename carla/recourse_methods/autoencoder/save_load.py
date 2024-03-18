import os
from util.settings import global_setting

def get_home(models_home=None):
    """Return a path to the cache directory for trained autoencoders.

    This directory is then used by :func:`save`.

    If the ``models_home`` argument is not specified, it tries to read from the
    ``CF_MODELS`` environment variable and defaults to ``~/cf-bechmark/models``.

    """

    if models_home is None:
        models_home = os.path.join(global_setting['home_directory'], "carla", "recourse_methods","autoencoder", "saved")
        
    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home
