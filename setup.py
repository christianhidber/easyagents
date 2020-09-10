import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easyagents",
    version="1.4.2",
    description="reinforcement learning for practitioners.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/christianhidber/easyagents",
    author="Christian Hidber, Oliver Zeigermann",
    author_email="christian.hidber@bsquare.ch",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["easyagents", "easyagents.callbacks", "easyagents.backends"],
    install_requires=[
        "tensorflow==2.3.0",
        "tensorflow-probability==0.11.0",
        "tf-agents==0.6.0",
        "tensorforce==0.6.0",
        "gym==0.17.2",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.2",
        "matplotlib==3.3.1"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "easyagents=easyagents.__main__:main",
        ]
    },
)
