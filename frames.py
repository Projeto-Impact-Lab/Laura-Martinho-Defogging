import cv2
import numpy as np

def add_fog(image, intensity=0.5):
    """
    Add fog to an image.

    Parameters:
    - image: The input image (numpy array).
    - intensity: The intensity of the fog effect (float between 0 and 1).

    Returns:
    - foggy_image: The image with added fog.
    """
    # Generate a greyish mask with the same data type as the input image
    mask = np.ones_like(image, dtype=np.float32) * (127 * intensity)

    # Blend the image with the mask to simulate fog
    foggy_image = cv2.addWeighted(image.astype(np.float32), 1 - intensity, mask, intensity, 0)

    return foggy_image.astype(np.uint8)

def processo(imagem):
    exposicao = 0.8
    contraste = 0.85
    brilho = 0.85
    image = enhance_segmentation(imagem)
    corrected = colorCorrection(image, 1.0)
    gamma = adjust_gamma(corrected,gamma=0.85)
    imagem_ajustada = ajustar_exposicao(gamma, exposicao, contraste, brilho)
    return imagem_ajustada

# Abre o vídeo usando cv2.VideoCapture()
video = cv2.VideoCapture("/content/video1ll.mp4")

# Verifica se o vídeo foi aberto com sucesso
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frames = []  # Lista para armazenar os frames do vídeo

# Loop para capturar cada frame do vídeo
while True:
    ret, frame = video.read()

    # Se não houver mais quadros para ler, saia do loop
    if not ret:
        break
    frame_processado = processo(frame)
    # Adiciona o frame à lista de frames
    frames.append(frame_processado)

# Libera o objeto de captura
video.release()

# Salva os frames como imagens individualmente
for i, frame in enumerate(frames):
    cv2.imwrite(f"/content/drive/MyDrive/Dehaze/HAU4/frame_{i}.png", frame)

print("Frames salvos como imagens individualmente.")

# Agora você pode usar os frames salvos para criar um novo vídeo, se necessário.
# Lista de frames salvos como imagens
frames = []
import os

# Diretório onde os frames foram salvos
diretorio_frames = "/content/drive/MyDrive/Dehaze/HAU4/"

# Lista todos os arquivos no diretório
arquivos_frames = os.listdir(diretorio_frames)

# Filtra os arquivos que correspondem ao padrão de nomenclatura dos frames
frames2 = [cv2.imread(os.path.join(diretorio_frames, arquivo)) for arquivo in arquivos_frames if arquivo.startswith("frame_")]

# Número de frames encontrados
num_frames = len(frames2)

print(f"Total de {num_frames} frames encontrados no diretório.")

# Carrega os frames das imagens
for i in range(num_frames):  # Substitua 'num_frames' pelo número real de frames salvos
    frame = cv2.imread(f"/content/drive/MyDrive/Dehaze/HAU9/frame_{i}.png")
    frames.append(frame)

# Cria um novo vídeo usando os frames
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
largura = frames[0].shape[1]  # Largura do quadro
altura = frames[0].shape[0]   # Altura do quadro
out = cv2.VideoWriter("/content/drive/MyDrive/Dehaze/HAU9/novo_video2.avi", fourcc, 20.0, (largura, altura))

# Adiciona os frames ao vídeo
for frame in frames:
    out.write(frame)

# Libera o objeto de escrita
out.release()

print("Novo vídeo criado com os frames salvos.")
