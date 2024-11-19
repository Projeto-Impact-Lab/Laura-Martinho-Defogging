import cv2
import numpy as np

def enhance_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masking = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    inv_masking = cv2.bitwise_not(masking)
    corrected_areas = mascaraNitidez(image, 1.0, 3.0, kernel=(5, 5), threshold=0)
    masked = cv2.bitwise_and(image, image, mask=inv_masking)
    masked += cv2.bitwise_and(corrected_areas, corrected_areas, mask=masking)
    return masked

# Carrega a imagem colorida
imagem = cv2.imread('/content/drive/MyDrive/HAZE/lanzhou_5.png')

enhanced = enhance_segmentation(imagem)
# Mostra a imagem com a máscara com o filtro aplicado
cv2_imshow(enhanced)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue sua imagem em tons de cinza
imagem_tons_de_cinza = cv2.imread('/content/drive/MyDrive/boa.jpg', cv2.IMREAD_GRAYSCALE)

# Normaliza os valores de tons de cinza para o intervalo [0, 1]
normalized_gray = cv2.normalize(imagem_tons_de_cinza, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Aplique a paleta de cores usando interpolação
thermal_map_resized = cv2.resize(normalized_gray, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)  # Ajuste a escala conforme necessário

# Crie um mapeamento de cores customizado (do frio para o quente)
# Neste exemplo, usei azul para frio, verde para morno e vermelho para quente
paleta_de_cores = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

# Aplique a paleta de cores manualmente
thermal_map_colored = cv2.applyColorMap((thermal_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Salve o mapa térmico colorido
plt.imshow(cv2.cvtColor(thermal_map_colored, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Oculta os eixos
plt.colorbar()
plt.show()

# Normaliza as temperaturas no mapa térmico para o intervalo [0, 1]
normalized_thermal_map = cv2.normalize(thermal_map_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Coeficiente de decaimento exponencial (ajuste conforme necessário)
beta = 0.1

# Estima o coeficiente de transmissão para cada pixel usando as temperaturas
transmission_map = np.exp(-beta * normalized_thermal_map)

# Exibe o mapa de transmissão
plt.imshow(transmission_map, cmap='gray')
plt.axis('off')
plt.colorbar()
plt.show()
