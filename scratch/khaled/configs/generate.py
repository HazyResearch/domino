## taken from Albert's hippo repo

# import argh
import argparse
import datetime

# import random
import importlib
import inspect
import itertools

# import os
import subprocess

import numpy as np

### These helpers return a list of lists
### Each list corresponds to one command line run, with a list of flags

chain = lambda l: list(itertools.chain(*l))


def _escape(k, v):
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return f"'{k}={v}'"
    else:
        return f"{k}={v}"


def flag(k, vs):
    """
    flag('seed', [0,1,2]) returns [['seed=0'], ['seed=1'], ['seed=2']]
    """
    return [[_escape(k, v)] for v in vs]


def pref(prefix, L):
    pref_fn = lambda s: prefix + "." + s
    # return map(functools.partial(map, pref_fn), prod(L))
    return [[pref_fn(s) for s in l] for l in prod(L)]


def prod(L):
    p = itertools.product(*L)
    return list(map(chain, p))


def lzip(L):
    if len(L) == 0:
        return []
    assert np.all(np.array(list(map(len, L))) == len(L[0])), "zip: unequal list lengths"

    out = map(chain, zip(*L))
    return list(out)


def cmdline(progname, name, args, dryrun=False):
    all_args = " ".join(args)
    if dryrun:
        # cmd = f"WANDB_MODE=dryrun python {progname}.py wandb.group=test {all_args} smoke_test=True"
        # cmd = f"WANDB_MODE=dryrun python {progname}.py wandb.group=test {all_args} runner.local=True"
        cmd = f"python -m {progname} wandb.group={name} {all_args}\n"
    else:
        cmd = f"python -m {progname} wandb.group={name} {all_args}\nsleep 10s"
    return cmd


def cmdfile(progname, name, sweep_fn, dryrun=False):
    # Timestamp
    timestamp = get_timestamp()
    print("timestamp", timestamp)
    run_name = timestamp
    if name is not None:
        run_name += name

    # Commit ID
    commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )

    # Generate runs
    runs = sweep_fn()

    # Print to file
    filename = f"configs/{run_name}.sh"
    try:
        with open(filename, "w") as cmdfile:
            cmdfile.write("#!/usr/bin/env bash\n\n")
            cmdfile.write(f"# {timestamp}\n")
            cmdfile.write(f"# {commit_id}\n\n")
            cmdfile.write(f"# {len(runs)} configurations\n\n")
            print(f"{len(runs)} configurations")
            for args in runs:
                cmd = cmdline(progname, run_name, args, dryrun)
                cmdfile.write(cmd + "\n")

            source = inspect.getsource(sweep_fn)
            commented_source = source.replace("\n", "\n# ")
            commented_source = "\n# " + commented_source
            cmdfile.write(commented_source)
    except:
        subprocess.run(["rm", filename])

    return filename


def get_timestamp():
    ts = datetime.datetime.now()
    # .replace(microsecond=0).isoformat()
    # print(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    ts_str = f"{ts:%Y-%m-%d-%H-%M-%S}"[2:]
    return ts_str


def default_config():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="train")
    parser.add_argument("--config", type=str)
    parser.add_argument("--sweep", default="sweep", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--dryrun", "-t", action="store_true")
    args = parser.parse_args()

    config = importlib.import_module(args.config)
    sweep_fn = getattr(config, args.sweep)
    f = cmdfile(args.file, args.name, sweep_fn, args.dryrun)
    if args.dryrun:
        subprocess.run(["cat", f])
        subprocess.run(["rm", f])
    else:
        subprocess.run(["chmod", "777", f])
        # subprocess.run(['./'+f])
