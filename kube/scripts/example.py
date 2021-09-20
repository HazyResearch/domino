import sys
import time
from collections import Counter

import ray


@ray.remote(
    runtime_env={"conda": "domino"},
)
def gethostname(x):
    import os
    import platform
    import sys
    import time

    import seaborn

    print(sys.executable)
    time.sleep(0.01)
    return x + (platform.node(),)


def wait_for_nodes(expected):
    # Wait for all nodes to join the cluster.
    while True:
        resources = ray.cluster_resources()
        node_keys = [key for key in resources if "node" in key]
        num_nodes = sum(resources[node_key] for node_key in node_keys)
        if num_nodes < expected:
            print(
                "{} nodes have joined so far, waiting for {} more.".format(
                    num_nodes, expected - num_nodes
                )
            )
            sys.stdout.flush()
            time.sleep(1)
        else:
            break


def main():
    wait_for_nodes(2)

    # Check that objects can be transferred from each node to each other node.
    for i in range(10):
        print("Iteration {}".format(i))
        results = [gethostname.remote(gethostname.remote(())) for _ in range(100)]
        print(Counter(ray.get(results)))
        sys.stdout.flush()

    print("Success!")
    sys.stdout.flush()


if __name__ == "__main__":
    print(sys.executable)
    print(ray.__version__)
    ray.init(
        "ray://10.96.15.32:10001",
    )
    main()
