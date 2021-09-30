import argparse


def parse_pipeline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_idx", type=int, help="The index of this worker.", default=None
    )
    parser.add_argument(
        "--num_workers", type=int, help="The total number of workers.", default=None
    )
    parser.add_argument("--synthetic", dest="synthetic", action="store_true")
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false")
    parser.set_defaults(synthetic=False)
    parser.add_argument("--subset", dest="synthetic", action="store_true")
    parser.add_argument("--no-subset", dest="synthetic", action="store_false")
    parser.set_defaults(synthetic=False)
    args = parser.parse_args()
    return args
