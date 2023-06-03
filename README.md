# SKUD - детекция и распознование лиц

## Подготовка

Для начала рекомендуется скачать несколько репозиториев для забора тестовых примеров скриптов, которые помогут в 
развитии проекта (для выполнения скриптов находиться нужно в корне проекта):

```commandline

wget https://github.com/ageitgey/face_recognition/archive/refs/heads/master.zip
unzip master.zip
rm master.zip

wget https://github.com/ultralytics/ultralytics/archive/refs/heads/main.zip
unzip main.zip
rm main.zip


```

## Структура платформы 

```commandline
faceRecognition
├── skud/
│   ├── __init__.py
│   ├── skud.py
│   ├── face_recognition.py
│   ├── feast_client.py
│   ├── models/
│   │   ├── yolov3.weights
│   │   ├── yolov3.cfg
│   │   └── face_recognition_model.pkl
│   ├── data/
│   │   └── face_database.csv
│   └── images/
│       ├── sample_image_1.jpg
│       └── sample_image_2.jpg
├── tests/
│   ├── __init__.py
│   ├── test_skud.py
│   ├── test_face_recognition.py
│   └── test_feast_client.py
├── README.md
├── requirements.txt
└── setup.py

```


- skud.py: Файл, содержащий главный класс Skud и код для запуска приложения Skud CLI.

- face_recognition.py: Модуль, содержащий код для детекции и распознавания лиц на изображении.

- feast_client.py: Модуль, содержащий код для взаимодействия с Feast Feature Store.

- models/: Каталог, где хранятся модели, необходимые для детекции и распознавания лиц.

- data/: Каталог, где хранятся данные, связанные с базой данных Feast.

- images/: Каталог, где хранятся примеры изображений для тестирования и демонстрации функций Skud.

- tests/: Каталог, содержащий модули тестирования для каждого модуля проекта Skud.

- README.md: Файл, содержащий описание и документацию проекта Skud.

- requirements.txt: Файл, содержащий список зависимостей проекта Skud.

- setup.py: Файл для установки и упаковки проекта Skud в виде Python-пакета.