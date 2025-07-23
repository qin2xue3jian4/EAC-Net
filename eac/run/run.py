import os
import sys
import torch
import psutil
import logging
import argparse
from typing import List, Dict
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils.envs import setup_seed
from ..data import get_loader, keys
from ..utils.version import SoftwareInfo
from ..model import get_model, BaseModel

def graph_to_labels(
    graph: Dict[str, Dict[str, torch.Tensor]],
    pred_keys: List[str],
):
    labels = {}
    for key, label in keys.LABELS.items():
        if key not in pred_keys:
            continue
        labels[key] = graph[label.parent][keys.REAL_PREFIX+label.key]
    return labels

@dataclass
class Runner:

    args: argparse.Namespace
    cfg: DictConfig

    def __post_init__(self):
        self.output_dir = HydraConfig.get().run.dir
        self._set_device()
    
    def _set_device(self):
        self.dtype = getattr(torch, f'float{self.args.dtype}')
        self.device = None
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ['RANK'])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.using_cuda = torch.cuda.is_available() and self.args.device != 'cpu'
        self.ddp_opening = self.world_size > 1

        if self.using_cuda:
            visible_rank = self.local_rank % torch.cuda.device_count()
            self.device = torch.device(f'cuda:{visible_rank}')
            self.device_ids = [visible_rank,]
        else:
            self.device = torch.device('cpu')
            self.device_ids = None

        if self.ddp_opening and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.device_count() >= self.world_size else 'gloo'
            dist.init_process_group(
                backend=backend,
                rank=self.local_rank,
                world_size=self.world_size
            )
        
        return
    
    def _log(self, msg: str, level=0, loglevel: str='INFO'):
        if self.local_rank == 0:
            log_level = getattr(logging, loglevel.upper(), logging.INFO)
            logging.log(log_level, '  '*level + msg)
        return
    
class Controller(Runner):
    def __post_init__(self):
        super().__post_init__()
        setup_seed(self.cfg.seed)
        self._print_base_infos()
        self._log(f'Output dir: {self.output_dir}')
        OmegaConf.save(self.cfg, os.path.join(self.output_dir, 'input.yaml'))
        self._load_model()
        self._get_out_type()
        self._log(f'Command argments: {" ".join(sys.argv[1:])}')
        
    def _print_base_infos(self):
        for line in SoftwareInfo.ascii_img.split('\n'):
            self._log(line)
        self._log(f'{SoftwareInfo.fullname}({SoftwareInfo.name}): v{SoftwareInfo.__version__}')
        self._log(SoftwareInfo.description)
        self._log('Hardware informations:')
        mem = psutil.virtual_memory()
        if self.using_cuda:
            self._log(f'using GPU to do task, num is {torch.cuda.device_count()}, world size is {self.world_size}.', 1)
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                msg = f'GPU {i}: {torch.cuda.get_device_name(i)}' + f', video memory: {memory:.2f} GB'
                self._log(msg, 1)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self._log(f'video memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved', 1)
        else:
            self._log(f'using CPU to do task.', 1)
        self._log(f'CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores.', 1)
        self._log(f'total memory: {mem.total / (1024.0 ** 3):.2f} GB', 1)
        return

    def _load_model(self):
        self._log('Building model.')
        
        # fine command model
        restored = self.args.model and os.path.exists(self.args.model)
        if restored:
            self._log(f'Resuming model from command model: {self.args.model}', 1)
            try:
                state_dict = torch.load(self.args.model, map_location='cpu', weights_only=True)
            except:
                state_dict = torch.load(self.args.model, map_location='cpu', weights_only=False)
            self.restored_cfg = OmegaConf.create(state_dict['settings'])
            if hasattr(self.cfg, 'model'):
                restored_model = state_dict['settings']['model']
                cfg_model = OmegaConf.to_container(self.cfg, resolve=True)['model']
                if restored_model != cfg_model:
                    self._log(f'model setting is different between input and restored model, using setting from input script.', 1, loglevel='warn')
            self.cfg = OmegaConf.merge(self.restored_cfg, self.cfg)
            method = 'command'
        else:
            if self.args.model:
                self._log(f'Model file {self.args.model} is not existing, building new model.', 1)
            method = 'new'
        
        # build
        self.module = model = get_model(self.cfg)
        model = model.to(device=self.device, dtype=self.dtype)
        if restored:
            finetune = hasattr(self.args, 'finetune') and self.args.finetune
            msgs = model.safely_load_state_dict(state_dict['model_state'], finetune)
            for msg in msgs:
                self._log(msg, 1, loglevel='warn')
        
        # fine checkpoint
        checkpoint = os.path.join(self.output_dir, 'models', 'checkpoint.pt')
        if not getattr(self.args, 'restart', False) and os.path.exists(checkpoint):
            try:
                state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
            except:
                state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict['model_state'])
            method = 'checkpoint'
        
        if hasattr(model, 'atom_env_irreps'):
            self._log(f'atom local environment irreps: {model.atom_env_irreps}', 1)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f'num of training params is {num_params}', 1)
        class_str = type(model)
        self._log(f'class of model is {class_str}', 1)
        
        if method != 'new':
            self.state_dict = state_dict
        self.method = method
        self._log(f'Model is built.')
        
        if self.ddp_opening:
            self._log(f'Distributed training is enabled, world size is {self.world_size}.')
            model = DDP(
                module=model,
                device_ids=self.device_ids,
            )
        
        self.model = model
        return
    
    def _get_out_type(self):
        if self.args.out_type is not None:
            assert self.args.out_type in ['probe', 'potential', 'mixed']
            self.out_type = self.args.out_type
        else:
            if hasattr(self.module, 'probe_fit_model'):
                if hasattr(self.module, 'potential_nets') and self.cfg.mode == 'train':
                    self.out_type = 'mixed'
                else:
                    self.out_type = 'probe'
            else:
                self.out_type = 'potential'
        self.spin = 2 if self.out_type != 'potential' and self.cfg.model.probe.spin else 1
        return
    
    def _load_paths_data(
        self,
        paths: List[str],
        frame_size: int = None,
        probe_size: int = None,
        num_workers: int = None,
        epoch_size: int = None,
    ):
        ngfs_str = self.args.ngfs if hasattr(self.args, 'ngfs') else None
        loader = get_loader(
            paths,
            self.args.mode,
            root_dir=self.cfg.data.root_dir,
            out_type=self.out_type,
            frame_size=frame_size,
            atom_rcut=self.cfg.data.atom_rcut,
            atom_sel=self.cfg.data.atom_sel,
            probe_size=probe_size,
            probe_rcut=self.cfg.data.probe_rcut,
            probe_sel=self.cfg.data.probe_sel,
            num_workers=num_workers,
            epoch_size=epoch_size,
            dtype=self.dtype,
            device=self.device,
            ngfs_str=ngfs_str,
            local_rank=self.local_rank,
            world_size=self.world_size,
            search_depth=self.args.search_depth,
            lazy_load=self.cfg.data.lazy_load,
            base_seed=self.cfg.seed,
        )
        return loader
    
    def run(self):
        if hasattr(self, 'out_type'):
            func = getattr(self, f'run_{self.out_type}')
            func()
        return