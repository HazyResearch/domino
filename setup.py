import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = [
    "pre-commit",
    "pytorch-lightning",
    "gradio<2.0.0",  # 1.7.7
    "terra @ git+https://github.com/seyuboglu/terra",
    "pandas",
    "numpy>=1.18.0",
    "cytoolz",
    "ujson",
    "jsonlines>=1.2.0",
    "torch>=1.8.0",
    "tqdm>=4.49.0",
    "scikit-learn",
    "umap-learn[plot]",
    "torchvision>=0.9.0",
    "wandb",
    "ray[default]",
    "torchxrayvision",
]

setuptools.setup(
    name="domino",
    version="0.0.1",
    author="Sabri Eyuboglu",
    author_email="eyuboglu@stanford.edu",
    description="Research package for automated slice discovery ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/domino",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=REQUIRED,
)
