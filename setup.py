from setuptools import setup

setup(
    name='socialchoice-kit',
    version='0.0.1',
    description='socialchoice-kit aims to be a comprehensive implementation of the most important rules in computational social choice theory.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Natsu Ozawa',
    author_email='natsuozawa@outlook.com',
    url='https://github.com/natsuozawa/socialchoice-kit',
    # TODO: set license
    # license='xxx',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # TODO: Add license classifier
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='computational social choice, social choice, voting, allocation, matching, algorithmic game theory, game theory',
    packages=['socialchoicekit'],
    install_requires=[
        'numpy',
        'scipy',
    ],
)
