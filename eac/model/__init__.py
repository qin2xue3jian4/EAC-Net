import pkgutil
import importlib
from typing import Union, Dict, cast
from omegaconf import (
    OmegaConf,
    DictConfig,
)
from .base_model import BaseModel, ModelFactory

for __, modulu_name, __ in pkgutil.iter_modules(__path__, prefix=__name__ + '.'):
    importlib.import_module(modulu_name)

DEFAULT_MODEL = 'eac'

def get_model(cfg: Union[DictConfig, Dict]) -> BaseModel:
    if isinstance(cfg, DictConfig):
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_config = cfg['model']
    model_config = cast(dict, model_config)
    model_type = model_config.pop('type', DEFAULT_MODEL)
    assert model_type in ModelFactory._registry, f'model type {model_type} not found'
    model = ModelFactory.create(model_type, **model_config)
    return model
