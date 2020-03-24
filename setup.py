import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='imsearch',
    version='0.1.1',
    description='A generic framework to build your own reverse image search engine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rikenmehta03/imsearch',
    author='Riken Mehta',
    author_email='riken.mehta03@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchvision', 'pandas', 'tables', 'redis', 'pymongo', 'nmslib', 'wget', 'opencv-python', 'requests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ]
)
