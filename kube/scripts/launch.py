import argparse
from pathlib import Path
from datetime import datetime
import tqdm
import yaml
import os
import subprocess



GPUS = {
    "v100": "nvidia-tesla-v100",
    "p100": "nvidia-tesla-p100",
    "a100": "nvidia-tesla-a100",
    "t4": "nvidia-tesla-t4",
}

JOB_PATH = {
    "gpu":"/Users/sabrieyuboglu/code/domino/kube/jobs/pool-gpu.yaml",
    "cpu": "/Users/sabrieyuboglu/code/domino/kube/jobs/pool-cpu.yaml"
} 
JOB_DIRECTORY = "/Users/sabrieyuboglu/code/domino/kube/jobs/tmp"


def create_job_config(
    script_path: str, job_name: str, worker_idx: int, num_workers: int, machine_type: str
):
    # Load up a base config
    base_config = yaml.load(open(JOB_PATH[machine_type]), Loader=yaml.FullLoader)

    # Modify it
    base_config["metadata"]["name"] = job_name
    base_config["spec"]["template"]["spec"]["containers"][0]["name"] = job_name

    # Add in the startup commands
    base_config["spec"]["template"]["spec"]["containers"][0]["command"] = [
        "/bin/bash",
        "-c",
    ]

    cmds = [
        "bash /pd/sabri/code/domino/kube/scripts/worker_startup.sh "
        f'"{script_path}" {worker_idx} {num_workers}',
    ]

    base_config["spec"]["template"]["spec"]["containers"][0]["args"] = [
        " && ".join(cmds)
    ]

    return base_config


def launch_kubernetes_job(path):
    # Execute a job
    subprocess.run(["kubectl", "create", "-f", f"{path}"])


def launch(args):
    """ To delete jobs afterwards, you can use. 
    ```
    `for i in `seq 0 5`; do k delete jobs.batch script-imagenet-worker$i; done
    ```
    """

    # Keep track of whatever job manifests (.yaml) we're generating
    # A single job will run a single configuration to completion
    job_yaml_paths = []

    # Go over each parameter configuration
    for worker_idx in range(args.num_workers):
        job_name = f"script-{Path(args.script_path).stem}-worker{worker_idx}"
        # Create a job configuration to run this
        config = create_job_config(
            script_path=args.script_path,
            job_name=job_name,
            worker_idx=worker_idx,
            num_workers=args.num_workers,
            machine_type=args.machine_type
        )
        datetime.today().strftime("%y-%m-%d")
        job_path = os.path.join(
            JOB_DIRECTORY, f"{datetime.today().strftime('%y-%m-%d')}_{job_name}.yaml"
        )
        yaml.dump(config, open(job_path, "w"))

        # Append to the queue of jobs we're running
        job_yaml_paths.append(job_path)

    # Launch all the Kubernetes jobs
    for path in tqdm.tqdm(job_yaml_paths):
        launch_kubernetes_job(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", "-s", type=str, required=True)
    parser.add_argument("--num_workers", "-n", type=int, required=True)
    parser.add_argument(
        "--machine_type", "-m", type=str, choices=["gpu", "cpu"], default="gpu"
    )
    args = parser.parse_args()
    launch(args)
