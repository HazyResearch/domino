#/bin/bash
export WANDB_API_KEY=394a25fdf10b50a3ac236f8db0ba6db8625d33e5

# link my code directory to my home 
ln -s /pd/sabri/code ~/code
ln -s /pd/* /home


source /home/common/envs/conda/bin/activate /home/common/envs/conda/envs/domino
python /pd/sabri/code/domino/scratch/sabri/09-21_celeba_pipeline.py