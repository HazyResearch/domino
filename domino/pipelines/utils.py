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
    parser.add_argument("--specific", dest="specific", action="store_true")
    parser.add_argument("--no-specific", dest="specific", action="store_false")
    parser.set_defaults(specific=False)
    parser.add_argument("--sanity", dest="sanity", action="store_true")
    parser.add_argument("--no-sanity", dest="sanity", action="store_false")
    parser.set_defaults(sanity=False)
    args = parser.parse_args()
    return args
