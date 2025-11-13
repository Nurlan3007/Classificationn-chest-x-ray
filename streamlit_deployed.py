import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # извлекаем признаки, испольузуем 3х3 фильтры потому-что он хорошо находит локальные особенности
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  
            nn.ReLU(), # фнкция активации нужна чтобы убрать линейность
            nn.MaxPool2d(2, 2), # уменьшение размерности
            nn.Dropout(0.25), # отключаем 1/4 нейронов для борьбы с переобучением

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # переводет многомерный тензор в одномерный тензор 
            nn.Linear(128 * 8 * 8, 256), # 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): # прямой проход через сеть
        x = self.features(x)
        x = self.classifier(x)
        return x

num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=num_classes).to(device)

best_path = "C:/Users/HP/Desktop/finalAML/best_model/best34.pth"

checkpoint = torch.load(best_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Resumed from epoch {checkpoint['epoch']}")

st.title("Классификация рентгена легких на 3 класса")
st.write("Загрузите изображение для предсказания диагноза.")

uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # размер, на котором обучалась модель
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # добавляем batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = ["Норм", "Пневмония", "Туберкулез"]  # замени на свои классы
    st.write(f"Предсказанный диагноз: **{classes[predicted.item()]}**")


