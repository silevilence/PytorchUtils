import json
import yaml
import sys
import os

if __name__ == '__main__':
    json_file = sys.argv[1]
    if len(sys.argv) > 2:
        yaml_file = sys.argv[2]
    else:
        yaml_file = f"{os.path.splitext(json_file)[0]}.yaml"

    yaml.dump(json.load(open(json_file, 'r')), open(yaml_file, 'w'))
