import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Загружаем обученную модель
model = tf.keras.models.load_model("animal_model.h5")

# Список классов (можешь заменить своим списком, если есть)
classes = [line.strip() for line in open("name of the animals.txt")]

# Функция для предсказания
def predict(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    return f"Это {classes[top_index]} ({preds[top_index]*100:.2f}% уверенности)"

# Интерфейс
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Результат"),
    title="Распознавание животных 🐶🐱🦁",
    description="Загрузите изображение животного — и модель скажет, кто это!"
)

if __name__ == "__main__":
    demo.launch()
