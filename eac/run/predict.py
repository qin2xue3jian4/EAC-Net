import os
import tqdm
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from .run import Controller
from ..data import keys
from ..data.write import Writer

class Predictor(Controller):
    def __post_init__(self):
        super().__post_init__()
        frame_size = 1 if self.out_type == 'probe' or self.args.frame_size is None else self.args.frame_size
        probe_size = self.args.probe_size or 50
        epoch_size = -1
        num_workers = self.args.num_workers or 0
        self.loader = self._load_paths_data(
            self.args.paths,
            frame_size=frame_size,
            probe_size=probe_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
        )
        self.model.eval()
    
    def run_probe(self):
        ifile, group_idx = 0, 0
        for istructure, (frame_id, igroup, iframe, result) in enumerate(self.inference_probe_loader(
            self.loader,
            need_contribute=self.args.contribute,
            need_label=False
        )):
            base_filename = self.generate_filename(frame_id, istructure)
            if group_idx == 0:
                writer = Writer()
            if iframe == 0:
                group_idx += 1
                space_group = self.loader.dataset.groups[igroup]
                natom = space_group.group[keys.ATOM_POS].shape[1]
                ngfs = self.loader.dataset.predict_ngfs[igroup]
                nframe = self.loader.dataset.nframes[igroup]
                frame_results = []
            frame_results.append(result)
            if iframe == nframe - 1:
                writer.append_result(frame_id, space_group, frame_results, sample_probe=False, ngfs=ngfs)

            if self.args.contribute:
                atom_contributions = result['atom_contributions'].view(natom, np.prod(ngfs), self.spin)
                for inode, node_value in enumerate(atom_contributions):
                    atom_preds = {}
                    for ispin in range(self.spin):
                        node_spin_value = node_value[:,ispin]
                        if self.spin == 2:
                            value_type = keys.CHARGE if ispin == 0 else keys.CHARGE_DIFF
                        else:
                            value_type = keys.CHARGE
                        atom_preds[value_type] = [node_spin_value]
                    file = os.path.join(self.output_dir, f'{base_filename}_atom_{inode}.chgcar')
                    writer.write_to_chgcar(file, atom_preds, iframe, ngfs)
            ngroup = self.loader.dataset.ngroups[ifile]
            if group_idx >= ngroup:
                base_filename = self.generate_filename(frame_id, istructure)
                file = os.path.join(self.output_dir, f'{base_filename}.{self.args.format}')
                writer.write_to_file(file)
                ifile += 1
                group_idx = 0
        return
