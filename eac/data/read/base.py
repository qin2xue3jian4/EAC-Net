from dataclasses import dataclass
from typing import Union, List, Dict

from ...utils.factory import BaseFactory
from .group import SpaceGroup, BaseGroup

class ReaderFactory(BaseFactory):
    _registry = {}

@dataclass
class BaseReader:
    file_path: str = None
    out_probe: bool = True
    lazy_load: bool = False
    
    def __post_init__(self):
        self.groups: List[Union[SpaceGroup, BaseGroup]] = []
        self.group_keys: List[str] = []
        if self.file_path is not None:
            groups_dict = self.load_file()
            for group_key, (group, extro_infos) in groups_dict.items():
                if self.out_probe:
                    space_group = SpaceGroup(group, extro_infos)
                else:
                    space_group = BaseGroup(group, extro_infos)
                self.group_keys.append(group_key)
                self.groups.append(space_group)
    
    def load_file(self):
        raise NotImplementedError
    
    def close(self):
        pass