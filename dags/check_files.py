import os
from pathlib import Path

print("current working directory", os.getcwd())
print('Contents of correct directory', os.listdir())
print('Contents of /opt/airflow', os.listdir('/opt/airflow'))
print('Contents of /opt/airflow/datasets', os.listdir('/opt/airflow/datasets'))

file_path='/opt/airflow/datasets/tracks.csv'
print("Does the file exist: ", Path(file_path).exists())