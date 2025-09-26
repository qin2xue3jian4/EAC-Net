from typing import Dict, Union, Type

from .run import Controller
from .train import Trainer
from .test import Tester
from .predict import Predictor

Runners: Dict[str, Type[Controller]] = {
    'train': Trainer,
    'test': Tester,
    'predict': Predictor,
}
