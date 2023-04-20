import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--json_path', type=str)
    args = parser.parse_args()

    fpath_list = sorted([os.path.join(args.data_path, fname) for fname in os.listdir(args.data_path) if fname.endswith('.pt')])
    os.makedirs(args.json_path, exist_ok=True)
    json.dump(fpath_list, open(args.json_path, 'w'))

