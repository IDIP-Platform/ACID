from pathlib import Path
from omegaconf import OmegaConf

config_path = Path("src/acid/config/default.yaml")

conf = OmegaConf.load(config_path)

if __name__ == "__main__":
    print(f"Config path: {config_path.resolve()}")

    print(OmegaConf.to_yaml(conf))

    print(conf.params_notebook_1.project.project_name)
