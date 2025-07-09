import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from eac.model import get_model
from eac.data import MixDataset, keys
from eac.data.load.collate import BatchCollater

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

def load_model(model_file):
    model_state = torch.load(model_file, map_location="cpu", weights_only=True)
    settings = model_state["settings"]
    model = get_model(settings)
    model.load_state_dict(model_state["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, settings

def load_data(data_file, settings, ngfs):
    atom_rcut = settings["data"]["atom_rcut"]
    atom_sel = settings["data"]["atom_sel"]
    probe_rcut = settings["data"]["probe_rcut"]
    probe_sel = settings["data"]["probe_sel"]
    predict_ngfs = ngfs
    dataset = MixDataset(
        data_file,
        mode='predict',
        out_type='mixed',
        root_dir=None,
        atom_cutoff=atom_rcut,
        atom_sel=atom_sel,
        probe_size=50,
        probe_cutoff=probe_rcut,
        probe_sel=probe_sel,
        predict_ngfs=predict_ngfs,
        dtype=dtype,
        search_depth=6,
        lazy_load=False,
    )
    return dataset

def collect_data(dataset, collater, idots: np.ndarray, iframe: int=0, igroup: int=0):
    if idots.ndim > 1:
        __, ngfs = group_info(dataset, igroup)
        igrids = np.ravel_multi_index(idots.T, ngfs) # (nxs, nys, nzs) -> (ms,)
    else:
        igrids = idots
    out_data = dataset.get_igroup_iframe_idots(igroup, iframe, igrids)
    out_data[keys.FRAME_ID] = f'{dataset.group_keys[igroup]}:{iframe}'
    batch = [out_data]
    data = collater(batch)
    newdata = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            newvalue = value.to(device=device)
        elif isinstance(value, dict):
            newvalue = {}
            for keykey, valuevalue in value.items():
                if isinstance(valuevalue, torch.Tensor):
                    newvaluevalue = valuevalue.to(device=device)
                else:
                    newvaluevalue = valuevalue
                newvalue[keykey] = newvaluevalue
        else:
            newvalue = value
        newdata[key] = newvalue
    return newdata

def group_info(dataset, igroup: int):
    group = dataset.groups[igroup]
    return group.nframe, dataset.predict_ngfs

def debug(model_file, dataset_file, ngfs):
    model, settings = load_model(model_file)
    dataset = load_data(dataset_file, settings, ngfs)
    collater = BatchCollater(dataset=dataset)
    ngroup = len(dataset.group_keys)
    igroup, iframe = 0, 0
    nprobe = 10
    x_values = np.ones(nprobe, dtype=int) * np.arange(0, nprobe, dtype=int)
    y_values = np.ones(len(x_values), dtype=int) * 0
    z_values = np.ones(len(x_values), dtype=int) * 0
    idots = np.column_stack((x_values, y_values, z_values))
    data = collect_data(dataset, collater, idots, iframe, igroup)
    preds = model(data)
    chgs = preds['chg'].detach().cpu().numpy()
    plt.cla()
    plt.plot(chgs, label='pred')
    plt.legend()
    plt.savefig('outputs/debug.png')
    plt.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ngfs', type=str, required=True)
    args = parser.parse_args()
    ngfs = np.array([int(x) for x in args.ngfs.split('*')])
    debug(args.model, args.dataset, ngfs)