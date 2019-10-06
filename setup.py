import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easyagents",
<<<<<<< dueling-dqn
    version="1.1.22",
=======
    version="1.1.21",
>>>>>>> master
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
    install_requires=["gast==0.2.2", "gym==0.10.11", "imageio==2.4.0", "imageio-ffmpeg==0.3.0",
                      "matplotlib==3.1.1", "numpy==1.17.0", "PILLOW==6.1.0", "pyglet==1.3.2",
                      "Keras==2.3.0", "keras-rl==0.4.2", "tensorforce==0.5.1",
                      "tensorflow-probability==0.7.0", "tensorflow-estimator==1.14.0",
                      "tf-agents-nightly==0.2.0.dev20190928", "tf-nightly==1.15.0.dev20190821"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "easyagents=easyagents.__main__:main",
        ]
    },
)
