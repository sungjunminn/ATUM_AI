from setuptools import setup, find_packages

setup_requires = [

]

install_requires = [
  'Flask',
  'Flask-Cors',
  'konlpy',
  'numpy',
  'pandas',
  'PyMySQL',
  'sklearn',
  'setuptools==58.2.0',
  'tensorflow==2.9.1',
  'keras==2.9.0',
  # Flask==2.2.2
  # Flask-Cors==3.0.10
  # Flask-SQLAlchemy==2.5.1
  # konlpy==0.6.0
  # mysql-connector-python==8.0.30
  # numpy==1.23.3
  # pandas==1.4.4
  # PyMySQL==1.0.2
  # sklearn==0.0
  # SQLAlchemy==1.4.40
]
setup(
  name='AI_FlaskServer',
  version='0.1',
  description='ChatBot ML Server',
  packages=find_packages(),
  install_requires = install_requires,
  setup_requires = setup_requires,
)