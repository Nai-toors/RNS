from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image

# Замените классы на свои
imageClassList = {'0': 'Кабриолет', '1': 'Пикап', '2': 'Седан'}

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    
    # Получаем класс и уверенность
    predicted_class, confidence = predictImageData('.'+filePathName)
    
    # Передаем оба значения в контекст шаблона
    context = {
        'scorePrediction': predicted_class,
        'confidencePercent': confidence
    }
    
    return render(request, 'scorepage.html', context)


def predictImageData(filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.Resampling.LANCZOS))
    # Путь к вашей ONNX модели
    sess = onnxruntime.InferenceSession(r'D:\studies\RNS\DZ1\lizards_CNN_MOBILENETV2_8167_unfreeze.onnx') 
    
    # Получаем выход модели
    outputs = sess.run(None, {'input': np.asarray([img]).astype(np.float32)})
    
    # Преобразуем выходные данные в вероятности с помощью softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    # Применяем softmax к выходным данным модели
    probabilities = softmax(outputs[0][0])
    
    # Находим индекс класса с максимальной вероятностью
    predicted_class_index = np.argmax(probabilities)
    
    # Получаем значение вероятности для предсказанного класса (в процентах)
    confidence = float(probabilities[predicted_class_index] * 100)
    
    # Получаем название класса
    predicted_class = imageClassList[str(predicted_class_index)]
    
    return predicted_class, confidence


