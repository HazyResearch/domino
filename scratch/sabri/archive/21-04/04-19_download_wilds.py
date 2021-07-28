root_dir = "/home/common/datasets"

import wilds

for dataset_name in ["fmow"]:
    dataset = wilds.get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)
