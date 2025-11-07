# app.py
import os
import sys
import time
from pathlib import Path

# --- Настройки ---
MODEL_FILENAME = "animal_model.h5"
# Можно задать через переменные окружения, либо заменить строкой с id:
# os.environ["MODEL_DRIVE_ID"] = "1aBcD_EfGhiJkLmNoPqRstuVWxyz"
MODEL_DRIVE_ID = os.environ.get("MODEL_DRIVE_ID", "1uROpYaZ3WXuR-6Z3frm1qVC54fRPL2Sq")

# Путь к файлу классов (если он лежит в репо, можно просто включить его)
CLASSES_FILE = "name of the animals.txt"

# --- Функция скачивания с Google Drive (использует gdown) ---
def download_from_drive(file_id: str, out_path: str, tries: int = 3):
    """
    Скачивает файл по file_id с Google Drive в out_path.
    Требует установленный пакет gdown.
    """
    try:
        import gdown
    except Exception as e:
        print("gdown не установлен. Устанавливаю...", file=sys.stderr)
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    for attempt in range(1, tries + 1):
        try:
            print(f"Скачивание модели (попытка {attempt})...", flush=True)
            gdown.download(url, out_path, quiet=False)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print("Файл успешно скачан.")
                return True
        except Exception as ex:
            print("Ошибка при скачивании:", ex)
        time.sleep(2)
    return False

# --- Убедимся, что модель существует локально, иначе скачиваем ---
MODEL_PATH = Path(MODEL_FILENAME)

if not MODEL_PATH.exists():
    if MODEL_DRIVE_ID == "YOUR_FILE_ID_HERE" or not MODEL_DRIVE_ID:
        raise RuntimeError(
            "MODEL_DRIVE_ID не указан. Поставьте реальный file id из Google Drive "
            "в переменную окружения MODEL_DRIVE_ID или замените в коде."
        )
    success = download_from_drive(MODEL_DRIVE_ID, str(MODEL_PATH))
    if not success:
        raise RuntimeError("Не удалось скачать модель с Google Drive. Проверьте доступность файла и id.")

# --- Загружаем модель ---
print("Загрузка модели в память...", flush=True)
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Модель загружена.", flush=True)
except Exception as e:
    print("Ошибка при загрузке модели:", e, file=sys.stderr)
    raise

# --- Загружаем список классов ---
if not os.path.exists(CLASSES_FILE):
    print(f"Файл {CLASSES_FILE} не найден. Убедитесь, что он в репозитории.", file=sys.stderr)
    classes = []
else:
    with open(CLASSES_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

if not classes:
    print("Список классов пуст или не найден. Предсказания будут возвращать индекс класса.", file=sys.stderr)

# --- Интерфейс Gradio ---
import numpy as np
from PIL import Image
import gradio as gr

INPUT_SIZE = (128, 128)  # должен совпадать с размером, на котором обучалась модель

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(image: Image.Image):
    if image is None:
        return "Нет изображения"
    x = preprocess_image(image)
    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = classes[top_idx] if top_idx < len(classes) else f"class_{top_idx}"
    return f"{label} ({confidence*100:.2f}%)"

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Animal Classifier (download model from Google Drive)",
    description="Загрузка модели при первом запуске из Google Drive. Убедитесь, что MODEL_DRIVE_ID задан."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
