from setuptools import setup, find_packages

setup(
    name='EcoStackML',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
        'xgboost'
    ],
    entry_points={
        'console_scripts': [
            'run-stacking=scripts.run_stacking:main',
        ],
    },
)
