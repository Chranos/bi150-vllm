from glob import glob

import json
import re


def should_ignore(weight_name, exact_ignore: list, regex_ignore: list):
    if any(weight_name.startswith(ig) for ig in exact_ignore):
        return True
    if any(p.match(weight_name) for p in regex_ignore):
        return True
    
    return False


def compile_ignore(ignore_list: list | None = None, 
                   extra_ignore_list: list | None = None):
    exact = set()
    regex = []

    for x in (ignore_list or []) + (extra_ignore_list or []):
        if x.startswith("re:"):
            regex.append(re.compile(x[3:]))
        else:
            exact.add(x)
    return exact, regex



def check(weight_names, ignore_list):
    exact_ignore, regex_ignore = compile_ignore(ignore_list)
    missing = []
    for wn in weight_names:
        if not should_ignore(wn, exact_ignore, regex_ignore):
            missing.append(wn)
    
    print(missing)


if __name__ == "__main__":
    import yaml

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--weight-map", type=str)
    parser.add_argument("--model-info", type=str)

    args = parser.parse_args()

    with open(args.model_info, "r") as f:
        cfg = yaml.safe_load(f)
    
    quant_config = cfg['quant']
    ignore_list = []
    ignore_list += (quant_config['ignore'] or [])
    ignore_list += (quant_config['class']['int4'] or [])
    ignore_list += (quant_config['class']['int8'] or [])

    with open(args.weight_map, "r") as f:
        weight_names = json.load(f)['weight_map'].keys()
    

    check(list(weight_names), ignore_list)
    


