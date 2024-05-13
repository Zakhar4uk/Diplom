#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS
import subprocess
import sys
# os.environ["OMP_NUM_THREADS"] = "1"


app = Flask(__name__)
CORS(app)
uploads = 'D:/proverka/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4'}
cmd = ['python', 'D:/proverka/vision/video_loop.py']
os.environ["OMP_NUM_THREADS"] = "4"

def allowed_file(filename):
    _, ext = os.path.splitext(filename)
    ext = ext[1:].lower()  # Убираем точку из расширения и приводим к нижнему регистру
    return ext in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_file(os.path.join(os.getcwd(), 'index.html'))

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Проверяем, был ли отправлен файл
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']
        
        # Проверяем, допустим ли формат файла
        if file and allowed_file(file.filename):
            # Сохраняем файл на сервере
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            
            # Проверяем, сохранен ли файл
            if os.path.exists(file_path):
                print(f"Файл {file.filename} успешно сохранен по пути {file_path}")
                # Открываем изображение и обрабатываем его
                cmd.append(uploads + file.filename)
                print(cmd)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                process.wait()
                print(process.returncode)
                processed_data = {"result": f"Обработка завершена c кодом {process.returncode}"}
                return jsonify(processed_data)
if __name__ == '__main__':
    app.run(debug=True)
