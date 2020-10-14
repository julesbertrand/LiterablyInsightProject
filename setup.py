from setuptools import setup

setup(
    name='litreading',
    version='0.0.1',
    author='Jules Bertrand',
    author_email='julesbertrand13@gmail.com',
    url='https://github.com/julesbertrand/litreading-insight-project.git',
    license='MIT license',
    packages=['litreading'],
    install_requires = [
      'numpy',
      'pandas',
      'joblib',
      'scikit-learn',
      'xgboost',
      'num2words',
      'matplotlib',
      'seaborn'
      ],
      zip_safe=False
)
