from domino.gcp import launch_pod

if __name__ == "__main__":
    launch_pod(
        "test2",
        pool="t4-1",
        cmd="python /home/sabri/code/domino/scratch/sabri/06-09_train.py",
        env="domino",
    )
