from argparse import Namespace
from omegaconf.dictconfig import DictConfig

from eac.run import Runners
from eac.utils import argment_parse

def main():
    @argment_parse()
    def run(args: Namespace, cfg: DictConfig):
        Runner = Runners[args.mode]
        runner = Runner(args, cfg)
        runner.run()
    run()

if __name__ == "__main__":
    main()