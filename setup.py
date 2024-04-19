from setuptools import setup

setup(
    name='socialchoicekit',
    version='1.0.0',
    description='socialchoicekit aims to be a comprehensive implementation of the most important rules in computational social choice theory.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Natsu Ozawa',
    author_email='natsuozawa@outlook.com',
    url='https://github.com/natsuozawa/socialchoicekit',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='computational social choice, social choice, voting, allocation, matching, algorithmic game theory, game theory',
    packages=['socialchoicekit'],
    install_requires=[
        'numpy',
        'scipy',
        'preflibtools',
    ],
)
