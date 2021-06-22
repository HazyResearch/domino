import importlib
import inspect
import os
import subprocess

import yaml


def get_commands(pool, cmd, env):
    return [
        "source /home/.bashrc",
        "source /home/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {env}",
        "bash /home/.wandb/auth",
        # 'eval `ssh-agent -s`',
        # 'ssh-add ~/.ssh/id_rsa',
        # 'git pull --rebase',
        cmd,
    ]


def launch_pod(run_name: str, pool: str, cmd: str, env: str, image: str = None):
    # Load the base manifest for launching Pods
    config = yaml.load(
        open("scratch/sabri/pods/default-pod.yaml"),
        Loader=yaml.FullLoader,
    )

    # Wipe out the GPU node selector
    config["spec"]["nodeSelector"] = {}
    # Specify the pool
    config["spec"]["nodeSelector"]["cloud.google.com/gke-nodepool"] = f"{pool}"
    # Request GPUs
    config["spec"]["containers"][0]["resources"] = {
        "limits": {"nvidia.com/gpu": pool.split("-")[-1]},
        "requests": {"nvidia.com/gpu": pool.split("-")[-1]},
    }

    # Set the name of the Pod
    config["metadata"]["name"] = config["spec"]["containers"][0]["name"] = run_name
    # Set the name of the image we want the Pod to run
    if image is not None:
        config["spec"]["containers"][0]["image"] = image

    # Put in a bunch of startup get_commands
    config["spec"]["containers"][0]["command"] = ["bash", "-c"]
    config["spec"]["containers"][0]["args"] = [
        " && ".join(get_commands(pool, cmd, env))
    ]

    # Store it
    yaml.dump(config, open("temp.yaml", "w"))

    # Log
    print(f"Run name: {run_name}")

    # Launch the Pod
    subprocess.call("kubectl apply -f temp.yaml", shell=True)

    # Clean up
    os.remove("temp.yaml")
