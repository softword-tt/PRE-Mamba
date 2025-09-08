import os
import argparse
import json
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

options_str = os.environ.get("OPTIONS", "{}")
options_dict = json.loads(options_str)
# 可以在调试时使用这个来设置 CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')  # 默认使用0,1,2,3 GPU

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Main Training Script")
    parser.add_argument("--config-file", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    # parser.add_argument("--options", type=str, default="", help="Additional options")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--weight", type=str, default=None, help="Path to model weights")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:12345", help="Distributed training URL")
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines for distributed training")
    parser.add_argument("--machine-rank", type=int, default=0, help="Rank of the current machine")
    
    args = parser.parse_args()

    cfg = default_config_parser(args.config_file, options_dict)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

if __name__ == "__main__":
    main()
