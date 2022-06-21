import setuptools

setuptools.setup(
    name="chromakey-tool",
    version="0.1",
    license='MIT',
    author="skysys",
    author_email="skysys@skysys.co.kr",
    description="chromakey dataset augmentation tool",
    long_description=open('README.md').read(),
    url="https://github.com/minwook96/chromakey-tool",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'click',
        'rich',
        'opencv-python'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)