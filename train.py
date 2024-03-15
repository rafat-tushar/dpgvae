import yaml
from dpg import DPGAE

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_cheetah.yml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    dpgae = DPGAE(cfg)
    dpgae.train()
