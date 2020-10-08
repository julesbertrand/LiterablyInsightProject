from setuptools import setup

setup(
    name="litreadingai",
    author="Jules Bertrand",
    author_email="julesbertrand13@gmail.com",
    url="https://github.com/julesbertrand/litreading-insight-project.git",
    packages=["litreading"],
    install_requires = [
      'numpy',
      'pandas',
      'joblib',
      'sklearn',
      'xgboost',
      'num2words',
      'matplotlib',
      'seaborn'
      ],
      zip_safe=False
)
