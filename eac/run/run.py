import os
import sys
import torch
import psutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union
from dataclasses import dataclass
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils.envs import setup_seed
from ..data import get_loader, keys, LoaderWrapper
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
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.using_cuda = torch.cuda.is_available() and self.args.device != 'cpu'
        self.ddp_opening = self.world_size > 1

        if self.using_cuda:
            if ':' in self.args.device:
                self.device = torch.device(self.args.device)
                self.device_ids = [int(self.args.device.split(':')[1]),]
            else:
                visible_rank = self.local_rank % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{visible_rank}')
                self.device_ids = [visible_rank,]
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
            self.device_ids = None

        if self.ddp_opening and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.device_count() >= self.world_size else 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method="env://",
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
        model = get_model(self.cfg)
        self.module = model = model.to(device=self.device, dtype=self.dtype)
        if restored:
            finetune = hasattr(self.args, 'finetune') and self.args.finetune
            model_state = state_dict.get('ema', state_dict['model_state'])
            msgs = model.safely_load_state_dict(model_state, finetune)
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
            self._log(f'Resuming model from checkpoint: {checkpoint}', 1)
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
        if self.ddp_opening:
            dist.destroy_process_group()
        return

    def inference_single_data(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
        result: Dict[str, torch.Tensor],
        atom_representations: torch.Tensor,
        space_keys: List[str],
        need_label: bool
    ):
        nprobe = data[keys.PROBE][keys.POS].shape[0]
        ngfs = data[keys.GLOBAL][keys.PROBE_GRID_NGFS][0].detach().cpu().numpy().astype(int)

        if atom_representations is not None:
            data[keys.ATOM][keys.FEATURES] = atom_representations
        
        probe_empty = data[keys.PROBE_EDGE_KEY][keys.INDEX].numel() == 0
        if probe_empty:
            preds = {}
            for key in space_keys:
                preds[key] = torch.zeros((nprobe,), dtype=self.dtype, device=self.device)
        else:
            preds = self.model(data, out_type=self.out_type, return_atom_features=True)
            if atom_representations is None:
                atom_representations = preds[keys.ATOM_FEATURES]
        
        # record preds, labels and grid idxs
        idxs = data[keys.PROBE][keys.IDXS]
        result['index'][idxs] += 1.0
        result[keys.PROBE_POS][idxs] = data[keys.PROBE][keys.POS]
        for key in preds:
            if key not in keys.LABELS:
                continue
            result[f'pred_{key}'][idxs] = preds[key].detach()
        if need_label:
            labels = graph_to_labels(data, preds.keys())
            for key in labels:
                result[f'label_{key}'][idxs] = labels[key].detach()
        grid_ptr = data[keys.PROBE][keys.IDXS][0]
        
        # record node contribution
        if 'atom_contributions' in result and not probe_empty:
            edge = data[keys.PROBE_EDGE_KEY]
            atom_index, probe_index = edge[keys.INDEX]
            grid_edge_scalar = edge[keys.CHARGE].detach()
            linear_indices = (atom_index * np.prod(ngfs) + probe_index + grid_ptr)
            expanded_indices = linear_indices.unsqueeze(-1).expand(-1, self.spin)
            flat_charge = torch.zeros_like(result['atom_contributions'])
            flat_charge.scatter_add_(0, expanded_indices, grid_edge_scalar)
            result['atom_contributions'] += flat_charge
        
        return result, atom_representations
    
    def inference_probe_loader(
        self,
        loader: LoaderWrapper,
        need_contribute: bool = False,
        need_label: bool = False,
    ):
        def sync_records(result: Dict[str, torch.Tensor]):
            if self.ddp_opening:
                for k, v in result.items():
                    v = v.contiguous()
                    dist.all_reduce(v)
            result['index'] = result['index'] > 0.5
            if self.local_rank == 0 and not torch.all(result['index']):
                old_result = result
                result = {}
                for key, value in old_result.items():
                    if key in ['index', 'atom_contributions']:
                        result[key] = value
                    else:
                        result[key] = value[result['index']]
            return result

        def init_result(igroup: int, iframe: int):
            nprobe = loader.dataset.nprobes[igroup]
            result = {
                'index': torch.zeros(nprobe, dtype=self.dtype, device=self.device),
                keys.PROBE_POS: torch.zeros((nprobe, 3), dtype=self.dtype, device=self.device)
            }
            for return_key in (['label', 'pred'] if need_label else ['pred']):
                for space_key in space_keys:
                    result[f'{return_key}_{space_key}'] = torch.zeros(nprobe, dtype=self.dtype, device=self.device)
            if need_contribute:
                natom = loader.dataset.groups[igroup].group[keys.ATOM_POS].shape[1]
                ngfs = loader.dataset.predict_ngfs[igroup]
                result['atom_contributions'] = torch.zeros((natom*np.prod(ngfs), self.spin), dtype=self.dtype, device=self.device)
            return result
        
        def update_tqdm(msg: str):
            pbar.clear()
            self._log(f'Processing {msg}')
            pbar.refresh()
        assert loader.frame_size == 1, "Only support frame_size == 1"

        space_keys = [keys.CHARGE]
        if self.spin == 2:
            space_keys.append(keys.CHARGE_DIFF)
        
        data = None
        pbar = tqdm(total=len(loader), desc="Processing", position=0)
        last_file = None
        ngroup = len(loader.dataset.group_keys)
        
        for igroup, group_key in enumerate(loader.dataset.group_keys):
            for iframe in range(loader.dataset.nframes[igroup]):

                working_frame_id = f'{group_key}:{iframe}'
                result = init_result(igroup, iframe)
                atom_representations = None
                filename = group_key.split(':')[0]
                
                if self.args.loglevel == 'file' and (last_file is None or last_file != filename):
                    update_tqdm(filename)
                    last_file = filename
                elif self.args.loglevel == 'group' and iframe == 0:
                    update_tqdm(group_key)
                elif self.args.loglevel == 'frame':
                    update_tqdm(working_frame_id)
                elif igroup % int(self.args.loglevel) == 0:
                    update_tqdm(f'Groups:{igroup}-{igroup+int(self.args.loglevel)}/{ngroup}, file: {filename}')
                
                while True:
                    if data is None:
                        try:
                            data = next(loader)
                            pbar.update(1)
                        except StopIteration:
                            break
                    if data[keys.GLOBAL][keys.FRAME_ID][0] != working_frame_id:
                        break
                    
                    result, atom_representations = self.inference_single_data(data, result, atom_representations, space_keys, need_label)
                    data = None

                synced_result = sync_records(result)
                if self.local_rank == 0:
                    yield working_frame_id, igroup, iframe, synced_result
        
        pbar.close()
        return
    
    def generate_filename(self, frame_id: str, istructure: int):
        input_file, group_key, iframe = frame_id.split(':')
        output_filename = self.args.output_fmt.format(
            filename=os.path.basename(input_file).split('.')[0],
            groupkey=group_key,
            iframe=iframe,
            istructure=istructure,
        )
        return output_filename