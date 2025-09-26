import os
import sys
import argparse
from argparse import Namespace
from functools import wraps
from omegaconf import OmegaConf
from hydra import main as hydra_main
from typing import Callable, Any, cast
from omegaconf.dictconfig import DictConfig

def create_parser():
    """
    Create command line argument parser.
    """
    parser = argparse.ArgumentParser(description='A tool to manage EAC-Net.')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(
        dest='mode', 
        required=True, 
        help='Available sub-commands'
    )
    
    # ==================== Base Arguments (shared by all commands) ====================
    base_parser = argparse.ArgumentParser(add_help=False)
    
    # Output and visualization parameters
    base_parser.add_argument(
        '--out', 
        help='Output directory for saving files'
    )
    base_parser.add_argument(
        '--plot', 
        action='store_true', 
        help='Whether to plot results'
    )
    
    # Model configuration parameters
    base_parser.add_argument(
        '--dtype', 
        type=int, 
        default=32, 
        choices=[16, 32, 64],
        help='Model computation precision (16/32/64-bit floating point)'
    )
    base_parser.add_argument(
        '--device', 
        default='auto', 
        help='Device to run model on (auto/cpu/cuda/cuda:0, etc.)'
    )
    
    # Data loading parameters
    base_parser.add_argument(
        '--frame-size', 
        type=int, 
        help='Frame size (batch size)'
    )
    base_parser.add_argument(
        '--probe-size', 
        type=int, 
        help='Number of probe samples'
    )
    base_parser.add_argument(
        '--num-workers', 
        type=int, 
        help='Number of worker processes for data loader'
    )
    
    # Training related parameters
    base_parser.add_argument(
        '--epoch-size', 
        type=int, 
        help='Number of training epochs'
    )
    base_parser.add_argument(
        '--out-type', 
        type=str, 
        help='Model output type'
    )
    base_parser.add_argument(
        '--search-depth', 
        type=int, 
        default=6, 
        help='Search depth for data files'
    )
    
    # ==================== Train Subcommand Arguments ====================
    train_parser = subparsers.add_parser(
        'train', 
        parents=[base_parser], 
        help='Train a model'
    )
    
    # Required arguments
    train_parser.add_argument(
        'config', 
        type=str, 
        help='Path to YAML format configuration file'
    )
    
    # Model parameters
    train_parser.add_argument(
        '-m', '--model', 
        type=str, 
        help='Path to load model file'
    )
    
    # Training mode parameters
    train_parser.add_argument(
        '--restart', 
        action='store_true', 
        help='Whether to restart training from scratch (overwrite existing model)'
    )
    train_parser.add_argument(
        '--finetune', 
        action='store_true', 
        help='Whether to finetune the model'
    )
    
    # ==================== Test Subcommand Arguments ====================
    test_parser = subparsers.add_parser(
        'test', 
        parents=[base_parser], 
        help='Test model performance'
    )
    
    # Required arguments
    test_parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=True, 
        help='Path to model file to test'
    )
    
    # Data input parameters
    test_parser.add_argument(
        '-p', '--paths', 
        type=str, 
        action='append', 
        help='Paths to test data (can be used multiple times for multiple paths)'
    )
    test_parser.add_argument(
        '--size', 
        type=int, 
        default=-1, 
        help='Size of test dataset (-1 means use all data)'
    )
    
    # Output control parameters
    test_parser.add_argument(
        '--format', 
        type=str, 
        default='h5', 
        help='Format of test data files'
    )
    test_parser.add_argument(
        '--split', 
        action='store_true', 
        help='Whether to show individual test results for each file'
    )
    test_parser.add_argument(
        '--save', 
        action='store_true', 
        help='Whether to save test results'
    )
    test_parser.add_argument(
        '--loglevel', 
        type=str, 
        default='group', 
        choices=['group', 'file', 'frame'],
        help='Log detail level'
    )
    test_parser.add_argument(
        '--output-fmt', 
        type=str, 
        default='{istructure}-{filename}-{groupkey}', 
        help='Output filename format (supports placeholders like {filename})'
    )
    
    # ==================== Predict Subcommand Arguments ====================
    predict_parser = subparsers.add_parser(
        'predict', 
        parents=[base_parser], 
        help='Predict charge density from structures using a model'
    )
    
    # Required arguments
    predict_parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=True, 
        help='Path to model file for prediction'
    )
    
    # Data input parameters
    predict_parser.add_argument(
        '-p', '--paths', 
        type=str, 
        action='append', 
        help='Paths to prediction data (can be used multiple times for multiple paths)'
    )
    
    # Output configuration parameters
    predict_parser.add_argument(
        '-s', '--ngfs', 
        type=str, 
        default='origin', 
        help='Grid dimensions (ngf parameters) for output charge density'
    )
    predict_parser.add_argument(
        '--format', 
        type=str, 
        default='chgcar', 
        choices=['chgcar', 'h5', 'npy'],
        help='Format of output charge density files'
    )
    predict_parser.add_argument(
        '-a', '--contribute', 
        action='store_true', 
        help='Whether to output contributions from each atom'
    )
    predict_parser.add_argument(
        '--loglevel', 
        type=str, 
        default='group', 
        choices=['group', 'file', 'frame'],
        help='Log detail level'
    )
    predict_parser.add_argument(
        '--output-fmt', 
        type=str, 
        default='{istructure}-{filename}-{groupkey}', 
        help='Output filename format (supports placeholders like {filename})'
    )
    
    return parser

def pre_parse_args():
    parser = create_parser()
    
    args, unknown_args = parser.parse_known_args()
    
    cli_args = []
    sys.argv = [sys.argv[0]]
    has_set_outdir = False
    for unknown_arg in unknown_args:
        if unknown_arg.startswith('hydra.run.dir'):
            has_set_outdir = True
        if unknown_arg.startswith('hydra'):
            sys.argv.append(unknown_arg)
        else:
            cli_args.append(unknown_arg)
    
    if not has_set_outdir and args.out:
        sys.argv.append(f'hydra.run.dir={args.out}')
    
    return args, cli_args

def argment_parse() -> Callable[[Callable[[Namespace, DictConfig], None]], Callable[[], None]]:
    original_argv = sys.argv.copy()
    args, cli_args = pre_parse_args()
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    mode = args.mode
    
    def decorator(func: Callable[[Namespace, DictConfig], None]) -> Callable[[], None]:
        @hydra_main(config_path=config_path, config_name=mode, version_base=None)
        @wraps(func)
        def wrapped(cfg: DictConfig) -> None:
            
            # config: default -> input script -> command argments
            if mode == 'train' and args.config is not None:
                user_cfg = OmegaConf.load(args.config)
                cfg = cast(DictConfig, OmegaConf.merge(cfg, user_cfg))
            
            if len(cli_args) > 0:
                cli_conf = OmegaConf.from_cli(cli_args)
                cfg = cast(DictConfig, OmegaConf.merge(cfg, cli_conf))
            
            sys.argv = original_argv
            
            return func(args, cfg)
        return wrapped
    return decorator
