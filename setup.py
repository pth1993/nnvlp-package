from setuptools import setup
from setuptools.command.install import install
import subprocess, shlex
import os


class Downloader(install):
      def run(self):
            print("Installing latest Lasagne version")
            subprocess.run("pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip", shell=True)
            print("Downloading NLTK data...")
            subprocess.run("python -m nltk.downloader all", shell=True)
            print("Downloading Word Embeddings...")
            current_dir = os.path.dirname(os.path.realpath(__file__))
            subprocess.run("bash embedding.sh", shell=True)
            install.run(self)


setup(name='nnvlp',
      version='0.1.7',
      description='Neural Network-Based Vietnamese Language Processing',
      url='http://github.com/pth1993/nnvlp-package',
      author='Hoang Pham',
      author_email='phamthaihoang.hn@gmail.com',
      license='MIT',
      packages=['nnvlp'],
      install_requires=[
          'pyvi', 'numpy', 'theano', 'nltk'
      ],
      dependency_links=['https://github.com/Lasagne/Lasagne/archive/master.zip'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      cmdclass={
        'install': Downloader,
      },
      zip_safe=False)
