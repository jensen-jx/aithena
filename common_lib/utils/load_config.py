import yaml, os
from typing import Dict, Any


def load_config() -> Dict[Any, Any]:
    config = None
    with open(os.environ['CONFIG_PATH']) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            raise(Exception(e))
    return config
