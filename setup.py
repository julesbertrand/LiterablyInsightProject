from setuptools import setup

setup(
    name="literacy-insight-project",
    author="Jules Bertrand",
    author_email="julesbertrand13@gmail.com",
    url="https://github.com/julesbertrand/literacy-assessment-insight-project.git",
    packages=["litscore"],
    install_requires = [
      'numpy',
      'pandas',
      'joblib',
      'sklearn',
      'xgboost',
      'num2words',
      'matplotlib',
      'seaborn',
      'logging',  # to be removed ?
      ],
      zip_safe=False
)
