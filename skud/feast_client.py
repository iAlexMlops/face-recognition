import pandas as pd


class FeastClient:
    def __init__(self, face_database_path):
        self.face_database = self.load_face_database(face_database_path)

    def load_face_database(self, face_database_path):
        # Загрузка базы данных лиц из файла CSV или другого источника данных
        # и создание DataFrame с информацией о лицах
        face_database = pd.read_csv(face_database_path)
        return face_database

    def query_face(self, face_encoding):
        # Запрос информации о лице с заданным вектором признаков лица из базы данных Feast
        face_info = self.face_database.loc[self.face_database['encoding'] == face_encoding]
        if not face_info.empty:
            face_name = face_info['name'].values[0]
            return face_name
        else:
            return "Unknown"
