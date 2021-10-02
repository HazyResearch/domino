#/bin/bash
export WANDB_API_KEY=f0ee50eee0be72160afe33e8f6f1a2ceaca4bf93
ln -s /pd/* /home


source /pd/common/envs/conda/bin/activate /pd/common/envs/conda/envs/domino-maya
cd /pd/maya/domino

python $1 --worker_idx=$2 --num_workers=$3
