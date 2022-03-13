# _*_ coding: utf-8 _*_
"""
Time:     2022/3/7 15:37
Author:   ZHANG Yuwei
Version:  V 0.1
File:     setup.py
Describe:
"""

import setuptools

# Reads the content of your README.md into a variable to be used in the setup below
# with open("./README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='supertld',
    version='0.0.1',
    description='SuperTLD: Detecting TAD-like domains from RNA-associated interactions',
    # long_description=long_description,  # loads your README.md
    # long_description_content_type="text/markdown",  # README.md is of type 'markdown'
    author='Yu Wei Zhang',
    author_email='ywzhang224@gmail.com',
    url='https://github.com/deepomicslab/SuperTLD',
    packages=setuptools.find_packages(),
    classifiers=[  # https://pypi.org/classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)