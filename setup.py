from setuptools import setup

setup(name='nnvlp',
      version='0.1',
      description='Neural Network-Based Vietnamese Language Processing',
      url='http://github.com/pth1993',
      author='Hoang Pham',
      author_email='phamthaihoang.hn@gmail.com',
      license='MIT',
      packages=['nnvlp'],
      install_requires=[
          'pyvi', 'numpy', 'theano', 'lasagne', 'nltk'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)