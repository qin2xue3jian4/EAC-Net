from dataclasses import dataclass
from typing import Union, List

# global
GLOBAL = 'global'
BATCH = 'batch'
CELL = 'cell'
POS = 'pos'
TYPE = 'type'
IDXS = 'idxs'
ATTR = 'attrs'
FEATURES = 'features'
FRAME_ID = 'frame_id'
INFOS = 'infos'
NUM_NODES = 'num_nodes'
SMOOTH = 'smooth'
WEIGHT_SMOOTH = 'weight_smooth'

# edge
INDEX = 'edge_index'
PERIODIC_INDEX = 'edge_periodic'
LENGTH = 'length'
BASIS_WEIGHT = 'basis_weight'
WEIGHT = 'weight'
EDGE_PERIODIC_INDEX = 'edge_periodic_index'
EDGE_ISOTROPIC_TYPE = 'bond'
EDGE_DIRECTIONAL_TYPE = 'to'
PERIODIC_VECTORS = 'periodic_vectors'

# atom
ATOM = 'atom'
ATOM_TYPE = f'{ATOM}_{TYPE}'
ATOM_POS = f'{ATOM}_{POS}'
ATOM_EDGE_INDEX = f'{ATOM}_{INDEX}'
ATOM_EDGE_KEY = (ATOM, EDGE_ISOTROPIC_TYPE, ATOM)
ATOM_ATTRS = f'{ATOM}_{ATTR}'
ATOM_FEATURES = f'{ATOM}_{FEATURES}'

# space
PROBE = 'probe'
PROBE_POS = f'{PROBE}_{POS}'
PROBE_GRID_SHAPE = f'{PROBE}_shape'
PROBE_GRID_NGFS = f'{PROBE}_ngfs'
PROBE_EDGE_INDEX = f'{PROBE}_{INDEX}'
PROBE_EDGE_KEY = (ATOM, EDGE_DIRECTIONAL_TYPE, PROBE)

# chgcar
FILE_SOURCE = 'file_source'
CHGCAR_AUG = 'aug'
CHGCAR_AUG_DIFF = 'augdiff'

# potential
DISPLACEMENT = 'displacement'

# label
REAL_PREFIX = 'real_'
PRED_PREFIX = 'pred_'
ENERGY = 'energy'
FORCE = 'force'
VIRIAL = 'virial'
CHARGE = 'chg'
CHARGE_DIFF = 'chgdiff'
CHARGE_UP = 'chgup'
CHARGE_DOWN = 'chgdown'

@dataclass
class Label:
    parent: str
    key: str
    simple: str
    progress: str
    probe: bool

GLOBAL_KEYS = {
    CELL: 'dtype',
    PROBE_GRID_NGFS: 'long',
    FRAME_ID: 'origin',
}

LABELS = {
    ENERGY: Label(GLOBAL, ENERGY, 'e', 'potential', False),
    FORCE: Label(ATOM, FORCE, 'f', 'force', False),
    VIRIAL: Label(GLOBAL, VIRIAL, 'v', 'virial', False),
    CHARGE: Label(PROBE, CHARGE, 'c', 'chg', True),
    CHARGE_DIFF: Label(PROBE, CHARGE_DIFF, 'd', 'chgdiff', True),
}
