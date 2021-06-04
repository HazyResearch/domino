from domino.data.iwildcam import build_iwildcam_df, iwildcam_task_config
from domino.vision import train

ROOT_DIR = "/afs/cs.stanford.edu/u/sabrieyuboglu/data/datasets"


train(
    data_df=build_iwildcam_df.out(282),
    max_epochs=20,
    batch_size=128,
    val_check_interval=0.05,
    **iwildcam_task_config
)
