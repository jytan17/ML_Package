import setuptools

setuptools.setup(
    name='models',
    version='0.0.1',
    author='Junyong Tan',
    author_email='jtan9801@gmail.com',
    description='A small machine learning package',
    url='https://github.com/jytan17/ML_Package',
    project_urls = {
        "Bug Tracker": "https://github.com/jytan17/ML_Package/issues"
    },
    license='MIT',
    packages=['models'],
    install_requires=['sklearn', 'numpy', 'cvxpy', 'matplotlib', 'scipy'],
)
