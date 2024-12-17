from setuptools import setup

setup(
    name="caption",
    version="0.1",
    packages=['caption'],
    install_requires=[
        'opencv-python==4.2.0.34',
        'torch==1.9.0',
        'torchvision==0.10.0',
        'Pillow==9.4.0',
        'scikit-learn',
        'wget',
        'docopt',
        'schema',
        'psutil',
        'loguru==0.5.2',
        'tqdm',
        'nltk',
        'jiwer',
        'argparse==1.4.0',
        'transformers==4.26.1',
        'loguru',
        'common_ml @ git+ssh://git@github.com/qluvio/common-ml.git#egg=common_ml',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py'
    ]
)