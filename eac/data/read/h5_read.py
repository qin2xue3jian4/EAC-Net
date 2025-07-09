import h5py
import logging

from typing import ClassVar, List
from .base import BaseReader, ReaderFactory

@ReaderFactory.register('h5')
class H5FileReader(BaseReader):

    patterns: ClassVar[List[str]] = ['*.h5']
    pattern_parent: ClassVar[bool] = False
    only_structure: ClassVar[bool] = False
    
    def load_file(self):
        f = h5py.File(self.file_path, 'r')
        groups_dict = {}
        for group_key, h5_group in f.items():
            array_group = {}
            for array_key, array_value in h5_group.items():
                array_group[array_key] = array_value[:] if not self.lazy_load else array_value
            attrs = {k: v for k, v in h5_group.attrs.items()}
            groups_dict[group_key] = (array_group, attrs)
        if self.lazy_load:
            self.f = f
        else:
            f.close()
        return groups_dict

    def close(self):
        if hasattr(self, 'f') and self.f is not None:
            try:
                self.f.close()
            except Exception as e:
                logging.info(f"Error closing file {self.file_path}: {e}")
        return
