sudo chmod o+rwx /pd
sudo chmod o+rwx /home


# link my code directory to my home 
ln -s /pd/sabri/code ~/code
ln -s /pd/* /home


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/common/envs/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/common/envs/conda/etc/profile.d/conda.sh" ]; then
        . "/home/common/envs/conda/etc/profile.d/conda.sh"
    else
        export PATH="/home/common/envs/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export RAY_CONDA_HOME="/pd/common/envs/conda/bin/conda"
export CONDA_EXE="/pd/common/envs/conda/bin/conda", 
