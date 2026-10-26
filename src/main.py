"""Preview a pipeline configuration without processing images."""

import argparse

from acid.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Override YAML; omit for defaults")
    parser.add_argument("--project-root", help="Base directory for relative paths")
    args = parser.parse_args()
    load_config(args.config, project_root=args.project_root)


if __name__ == "__main__":
    main()
