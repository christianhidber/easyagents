import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easyagents",
    version="0.0.28",
    description="Easy, simple and (hopefully) painless use of reinforcement learning algorithms (prototype / proof of concept)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/christianhidber/easyagents",
    author="Christian Hidber, ...",
    author_email="christian.hidber@bsquare.ch",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["easyagents"],
    install_requires=["gym==0.10.11", "imageio==2.4.0", "imageio-ffmpeg==0.3.0", "matplotlib==3.1.1", "PILLOW==6.1.0",
                      "pyglet==1.3.2", "tf-agents-nightly", "tf-nightly"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "easyagents=easyagents.__main__:main",
        ]
    },
)
