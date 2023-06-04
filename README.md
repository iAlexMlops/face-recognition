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
faceRecognition/
│
├── skud/                    # Main package
│   ├── __init__.py
│   ├── skud.py              # Main class and CLI app
│   ├── face_recognition.py  # Face detection and recognition code
│   ├── feast_client.py      # Code to interact with Feast Feature Store
│   ├── models/              # Directory for models used in face detection and recognition
│   ├── data/                # Directory for data related to Feast database
│   └── images/              # Directory for sample images for testing and demonstrating Skud functions
│
├── tests/                   # Directory for testing modules for each project module
│   ├── __init__.py
│   ├── test_skud.py
│   ├── test_face_recognition.py
│   └── test_feast_client.py
│
├── docs/                    # Directory for documentation
│   └── README.md
│
├── resources/               # Directory for additional resources (scripts, datasets, etc.)
│
├── .gitignore
├── Makefile
├── docker-compose.yaml
├── generate_dataset.py
├── macOS_test_gpu.py
├── main.py
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