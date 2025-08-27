import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
        
import argparse
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def infer_type(val):
    if isinstance(val, bool):
        return "flag"
    return type(val)

def parse_args_with_config(config_path="config.yaml"):
    # Load config entries
    config_data = load_config(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default="", help="User query")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")

    # Dynamically add optional overrides based on config entries
    for key, val in config_data.items():
        arg_type = infer_type(val)
        if arg_type == "flag":
            # Example: --rerank or not
            parser.add_argument(f"--{key}", action="store_true", help=f"(bool flag from config)")
        else:
            parser.add_argument(f"--{key}", type=arg_type, help=f"(override for {key})")

    args = parser.parse_args()

    # Re-load config now that we know which file might have been passed via CLI
    final_config = load_config(args.config)

    # Override with args (if not None)
    for key in final_config.keys():
        val = getattr(args, key, None)
        if val is not None:
            final_config[key] = val

    return args, Config(**final_config)