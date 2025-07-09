import os
import sys
import argparse
from functools import wraps
from omegaconf  import OmegaConf
from hydra import main as hydra_main
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

def pre_parse_args():
    parser = argparse.ArgumentParser(description='A tool to manager deep models.')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='sub commands')
    
    base_parser = argparse.ArgumentParser(add_help=False)
    # base_parser.add_argument('--out', default='outs', help='plot the result.')
    base_parser.add_argument('--plot', action='store_true', help='plot the result.')
    base_parser.add_argument('--dtype', type=int, default=32, help='model precision')
    base_parser.add_argument('--device', default='auto', help='model device')
    base_parser.add_argument('--frame-size', type=int, help='The size of frame.')
    base_parser.add_argument('--probe-size', type=int, help='The size of probe.')
    base_parser.add_argument('--num-workers', type=int, help='The num workers of dataloader.')
    base_parser.add_argument('--epoch-size', type=int, help='The epoch size.')
    base_parser.add_argument('--out-type', type=str, help='Model use types.')
    base_parser.add_argument('--search-depth', type=int, default=6, help='The depth of search data file.')
    
    train_parser = subparsers.add_parser('train', parents=[base_parser], help='Train a model.')
    train_parser.add_argument('config', type=str, help='A Yaml-format configuration file.')
    train_parser.add_argument('-m', '--model', type=str, help='Model file path.')
    train_parser.add_argument('--restart', action='store_true', help='Whether restart the training.')
    train_parser.add_argument('--finetune', action='store_true', help='Whether finetune the model.')
    
    test_parser = subparsers.add_parser('test', parents=[base_parser], help='Test a model.')
    test_parser.add_argument('-m', '--model', type=str, required=True, help='Model file path.')
    test_parser.add_argument('-p', '--paths', type=str, action='append', help='The paths for test.')
    test_parser.add_argument('--size', type=int, default=-1, help='The total size for test.')
    test_parser.add_argument('--format', type=str, default='npy', help='The total size for test.')
    test_parser.add_argument('--split', action='store_true', help='Show each file test result.')
    
    predict_parser = subparsers.add_parser('predict', parents=[base_parser], help='Apply a model to predict charge density from structures.')
    predict_parser.add_argument('-m', '--model', type=str, required=True, help='Model file path.')
    predict_parser.add_argument('-p', '--paths', type=str, action='append', help='The paths for predict.')
    predict_parser.add_argument('-s', '--ngfs', type=str, required=True, help='The ngfs of output chg.')
    predict_parser.add_argument('--format', type=str, default='chgcar', help='The final format of output chg.')
    predict_parser.add_argument('-a', '--contribute', action='store_true', help='Whether to output the contributions of each atom.')
    
    args, unknown_args = parser.parse_known_args()
    
    cli_args = []
    sys.argv = [sys.argv[0]]
    for unknown_arg in unknown_args:
        if 'hydra' in unknown_arg:
            sys.argv.append(unknown_arg)
        else:
            cli_args.append(unknown_arg)
    
    return args, cli_args

def argment_parse() -> callable:
    original_argv = sys.argv.copy()
    args, cli_args = pre_parse_args()
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    mode = args.mode
    
    def decorator(func):
        @hydra_main(config_path=config_path, config_name=mode, version_base=None)
        @wraps(func)
        def wrapped(cfg):
            
            if mode == 'train' and args.config is not None:
                user_cfg = OmegaConf.load(args.config)
                cfg = OmegaConf.merge(cfg, user_cfg)
            
            if len(cli_args) > 0:
                cli_conf = OmegaConf.from_cli(cli_args)
                cfg = OmegaConf.merge(cfg, cli_conf)
            
            sys.argv = original_argv
            
            return func(args, cfg)
        return wrapped
    return decorator
