import os
import glob
import fnmatch
import pkgutil
import importlib
from typing import List, Dict, Union

from .group import SpaceGroup, BaseGroup
from .base import BaseReader, ReaderFactory

for __, modulu_name, __ in pkgutil.iter_modules(__path__, prefix=__name__ + '.'):
    if modulu_name.endswith('_read'):
        importlib.import_module(modulu_name)

def _clean_paths(
    origin_paths: Union[str, List[str]],
    root_dir: str = None,
):
    if isinstance(origin_paths, str):
        origin_paths = [origin_paths,]
    
    file_paths = []
    for origin_path in origin_paths:
        if not os.path.exists(origin_path) and len(glob.glob(origin_path)) == 0:
            assert root_dir is not None, f"File not found: {origin_path}"
            origin_path = os.path.join(root_dir, origin_path)
        if os.path.exists(origin_path):
            file_paths.append(origin_path)
        elif len(glob.glob(origin_path)) > 0:
            file_paths.extend(glob.glob(origin_path))
        else:
            raise FileNotFoundError(f"File not found: {origin_path}")
    return file_paths

def file_path_to_mode_paths(
    file_path: str,
    search_depth: int,
    only_structure: bool = False,
):
    mode_paths = {}
    for reader_mode, reader_class in ReaderFactory._registry.items():
        
        if reader_class.only_structure != only_structure:
            continue
        
        for pattern in reader_class.patterns:
            if os.path.isfile(file_path):
                if fnmatch.fnmatch(os.path.basename(file_path), pattern):
                    real_path = os.path.dirname(file_path) if reader_class.pattern_parent else file_path
                    mode_paths[real_path] = reader_mode

            if os.path.isdir(file_path):
                for depth in range(search_depth):
                    path_pattern = os.path.join(file_path, *(['*']*depth), pattern)
                    for path in glob.glob(path_pattern):
                        real_path = os.path.dirname(path) if reader_class.pattern_parent else path
                        mode_paths[real_path] = reader_mode
            
    return mode_paths

def file_paths_to_reader_modes(
    file_paths: List[str],
    root_dir: str = None,
    search_depth: int = 6,
):
    clean_paths = _clean_paths(file_paths, root_dir)
    
    real_paths: Dict[str, str] = {}
    for file_path in clean_paths:
        
        mode_paths = file_path_to_mode_paths(file_path, search_depth, only_structure=False)
        
        if len(mode_paths) == 0:
            mode_paths = file_path_to_mode_paths(file_path, search_depth, only_structure=True)
        
        assert len(mode_paths) > 0, f'No reader found for {file_path}'
        
        real_paths.update(mode_paths)
    
    # sort
    real_paths = {
        k: v
        for k, v in sorted(real_paths.items(), key=lambda x: x[0])
    }
    
    return real_paths

def file_paths_to_readers(
    file_paths: Union[List[str], str],
    out_probe: bool = True,
    root_dir: str = None,
    lazy_load: bool = False,
    search_depth: int = 6,
):
    path_modes = file_paths_to_reader_modes(file_paths, root_dir, search_depth)
    readers: Dict[str, BaseReader] = {}
    for real_path, reader_mode in path_modes.items():
        readers[real_path] = ReaderFactory.create(
            reader_mode,
            file_path=real_path,
            out_probe=out_probe,
            lazy_load=lazy_load
        )
    
    return readers