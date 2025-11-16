import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


# --- 1. Definición del Modelo ---
# <-- CAMBIO: Esta es la CNNNet_v3 que coincide con tu modelo entrenado
class CNNNet_v3(nn.Module):
    def __init__(self):
        super(CNNNet_v3, self).__init__()

        # --- Bloques Convolucionales ---
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        # --- CAPA ADAPTATIVA ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # --- Capas Densas ---
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4) # 4 clases

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Aplicamos los bloques CONV -> BN -> RELU -> POOL
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # --- Aplicamos la capa adaptativa ANTES de aplanar ---
        x = self.adaptive_pool(x)

        # Aplanamos
        x = x.view(x.size(0), -1)

        # Capas densas
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# --- 2. Configuración de Inferencia ---
device = torch.device('cpu')

# <-- CAMBIO: Apunta a tu nuevo modelo V3
MODEL_PATH = 'cnn_model_v3_4clases.pth'

# ¡IMPORTANTE! Asegúrate que el orden de esta lista coincida
# con el `train_data.class_to_idx` de tu Colab
labels = ['Diego', 'Luis', 'Medalith', 'Romulo']

# Cargar el modelo
print("Cargando modelo CNNNet_v3 (4 clases, 224px)...")
model = CNNNet_v3().to(device) # <-- CAMBIO: Carga la clase V3
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Poner en modo de evaluación
print("Modelo cargado exitosamente.")

# <-- CAMBIO: Transformaciones DEBEN coincidir con el entrenamiento (224px)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Inicializar la API de Flask ---
app = Flask(__name__)
CORS(app)  # Habilitar CORS


# --- 4. Endpoints ---
@app.route("/")
def home():
    return "API de Reconocimiento V3 (4 clases, 224px) funcionando."


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    try:
        # Leer la imagen
        contents = file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Aplicar transformaciones (ahora de 224px)
        x = img_transforms(img).unsqueeze(0).to(device)

        # Hacer predicción
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, 0)
            pred_clase = labels[idx.item()]
            confianza = conf.item()

        # --- Lógica de Umbral ---
        # Mantuvimos tu umbral del 90% para ser más riguroso
        UMBRAL_CONFIANZA = 0.90

        if confianza < UMBRAL_CONFIANZA:
            pred_clase = "Persona Desconocida"
        # --- Fin de la lógica ---

        return jsonify({"prediccion": pred_clase, "confianza": confianza})

    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


# --- 5. Iniciar el servidor ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)