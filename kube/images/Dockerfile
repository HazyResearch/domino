FROM  rayproject/ray:nightly-py38

ENV CONDA_EXE=/pd/common/envs/conda/bin/conda
ENV RAY_CONDA_HOME=/pd/common/envs/conda

RUN echo 'export PATH=/pd/common/envs/conda/bin:$PATH' >> /home/ray/.bashrc
RUN echo 'export CONDA_EXE=/pd/common/envs/conda/bin/conda' >> /home/ray/.bashrc