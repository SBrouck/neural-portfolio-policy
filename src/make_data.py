import argparse
from src.data_ingest import run_data_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="meta/universe.yaml")
    parser.add_argument("--config",   default="configs/data.yaml")
    args = parser.parse_args()
    run_data_pipeline(universe_yaml=args.universe, data_yaml=args.config)

if __name__ == "__main__":
    main()

