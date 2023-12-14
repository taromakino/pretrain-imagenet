from setuptools import setup, find_packages


setup(
    name='pretrain-imagenet',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'pytorch-lightning',
        'torch',
        'torchvision'
    ]
)