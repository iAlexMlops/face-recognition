from setuptools import setup, find_packages

setup(
    name='skud',
    version='1.0',
    description='Skud - Face Detection and Recognition System',
    author='Alex Egorov',
    author_email='ialex.egorov.mlops@gmail.com',
    packages=find_packages(),
    install_requires=[
        'Click>=6.0',
        'numpy',
        'Pillow',
        'scipy>=0.17.0',
        'ultralytics',
        'face_recognition==1.3.0',
        'dlib==19.22.0',
        'face_recognition_models',
        'feast',
        'apache-airflow==2.6.1',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'skud = skud.skud:main',
        ],
    },
)
