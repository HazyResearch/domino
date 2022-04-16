import io
import os
import sys
from distutils.util import convert_path
from shutil import rmtree

from setuptools import Command, find_packages, setup

main_ns = {}
ver_path = convert_path("domino/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


# Package meta-data.
NAME = "domino"
DESCRIPTION = ""
URL = ""
EMAIL = "eyuboglu@stanford.edu"
AUTHOR = "https://github.com/HazyResearch/domino"
REQUIRES_PYTHON = ">=3.8.6"
VERSION = main_ns["__version__"]

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = [
    "meerkat-ml[ml]",
    "pandas",
    "numpy>=1.18.0",
    "tqdm>=4.49.0",
    # TODO: support scikit-learn 1.0.0
    "scikit-learn==0.24.2",
    "ipywidgets",
    "seaborn",
    "torch",
    "torchvision",
    "ftfy",
    "regex",
]

EXTRAS = {
    "dev": [
        "black==21.5b0",
        "isort>=5.7.0",
        "autoflake",
        "flake8>=3.8.4",
        "mypy>=0.9",
        "docformatter>=1.4",
        "pytest-cov>=2.10.1",
        "sphinx-rtd-theme>=0.5.1",
        "nbsphinx>=0.8.0",
        "recommonmark>=0.7.1",
        "parameterized",
        "pre-commit>=2.9.3",
        "sphinx-autobuild",
        "furo",
    ],
    "text": ["transformers", "nltk"],
    "eval": ["pytorch-lightning", "dcbench"],
}


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
