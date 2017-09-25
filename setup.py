from setuptools import setup
from setuptools.command.install import install
import subprocess, shlex


class NLTKDownloader(install):
    """Post-installation for installation mode."""
    def run(self):
        subprocess.call(shlex.split("python -m nltk.downloader all"))
        install.run(self)


setup(name='nnvlp',
      version='0.1.1',
      description='Neural Network-Based Vietnamese Language Processing',
      url='http://github.com/pth1993/nnvlp-package',
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
      cmdclass={
        'install': NLTKDownloader,
      },
      zip_safe=False)
