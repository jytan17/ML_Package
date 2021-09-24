from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='models',
    url='https://github.com/jytan17/ML_Package',
    author='Junyong Tan',
    author_email='jtan9801@gmail.com',
    # Needed to actually package something
    packages=['models'] #, 'models.supervised', 'models.unsupervised'],
    # Needed for dependencies
    install_requires=['numpy', 'matplotlib', 'sklearn', 'cvxpy'],
    # The license can be anything you like
    license='None',
    description='A small package of commonly used machine learning algorithms'
)
