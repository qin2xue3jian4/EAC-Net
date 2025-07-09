from typing import Dict, Union

from .train import Trainer
from .test import Tester
from .predict import Predictor

Runners: Dict[str, Union[Trainer, Tester, Predictor]] = {
    'train': Trainer,
    'test': Tester,
    'predict': Predictor,
}
