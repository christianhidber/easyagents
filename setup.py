import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easy_agents",
    version="0.0.1",
    description="Easy, simple and (hopefully) painless use of reinforcement learning algorithms (prototype / proof of concept)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/johnbuild/EasyAgents",
    author="Christian Hidber, ...",
    author_email="christian.hidber@bsquare.ch",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["easy_agents"],
    install_requires=["gym==0.10.11","imageio==2.4.0","imageio-ffmpeg","matplotlib","PILLOW","pyglet","tf-agents-nightly","tf-nightly"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "easyagents=easy_agents.__main__:main",
        ]
    },
)