from setuptools import setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='TorchCRF',
    version='1.0.3',
    description='An Implementation of Conditional Random Fields in pytorch',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Ryuya Ikeda',
    author_email='rikeda71@gmail.com',
    install_requires=_requires_from_file('requirements.txt'),
    url='https://github.com/s14t284/TorchCRF',
    license='MIT',
    keywords=['crf', 'conditional random fields',
              'nlp', 'natural language processing'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing',
    ],
    packages=['TorchCRF'],
    test_suite='tests',
)
