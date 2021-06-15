from domino.gcp import launch_pod

if __name__ == "__main__":
    launch_pod(
        "test",
        pool="t4-1-train",
        cmd="python /home/sabri/code/domino/scratch/sabri/06-09_train.py",
        env="domino",
    )
