def enhance_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masking = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    inv_masking = cv2.bitwise_not(masking)
    corrected_areas = mascaraNitidez(image, 1.0, 2.0, kernel=(5, 5), threshold=0)
    masked = cv2.bitwise_and(image, image, mask=inv_masking)
    masked += cv2.bitwise_and(corrected_areas, corrected_areas, mask=masking)
    return masked


def colorCorrection(imagem, intensidade):
  resultados = [] #vetor para receber os resultados das trasnformações
  rgb = cv2.split(imagem) #acesso a cada canal de cor
  saturacao = rgb[0].shape[0] * rgb[0].shape[1] * intensidade / 500.0 #200
  for canal in rgb:
      histograma = cv2.calcHist([canal], [0], None, [256], (0,256), accumulate=False)
      #low value
      lowvalue = np.searchsorted(np.cumsum(histograma), saturacao) #soma acumulada dos elementos valor inferior do histograma e encontra índices onde os elementos devem ser inseridos p/ ordenar
      #high value
      highvalue = 255-np.searchsorted(np.cumsum(histograma[::-1]), saturacao)#soma acumulada e sort valores superiores
      #tomar toda a informação (min/max) da curva linear para aplicar e gerar uma LUT de 256 valores a aplicar nos canais stretching
      lut = np.array([0 if i < lowvalue else (255 if i > highvalue else round(float(i-lowvalue)/float(highvalue-lowvalue)*255)) for i in np.arange(0, 256)], dtype="uint8")
      #mescla os canais de volta
      resultados.append(cv2.LUT(canal, lut))
  return cv2.merge(resultados)

# Função para ajustar o contraste e a luminosidade da imagem
def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def mascaraNitidez(imagem, sigma, intensidade, kernel=(5, 5), threshold=0): #sigma 1.0/ 0.5, intensidade 2.0/ 1.0
  suavizacao = cv2.GaussianBlur(imagem, kernel, sigma)
  nitidez = float(intensidade + 1) * imagem - float(intensidade) * suavizacao
  nitidez = np.maximum(nitidez, np.zeros(nitidez.shape))
  nitidez = np.minimum(nitidez, 255 * np.ones(nitidez.shape))
  nitidez = nitidez.round().astype(np.uint8)
  if threshold > 0:
      contraste_baixo = np.absolute(imagem - suavizacao) < threshold
      np.copyto(nitidez, imagem, where=contraste_baixo)
  return nitidez


def ajustar_exposicao(imagem, exposicao, contraste, brilho):
    # Ajusta a exposição
    imagem_ajustada = np.clip(imagem * exposicao, 0, 255)
    # Ajusta o contraste e brilho
    imagem_ajustada = np.clip((imagem_ajustada - 127) * contraste + 127 + brilho, 0, 255)
    return imagem_ajustada.astype(np.uint8)

# imagem2 = enhanced
# z = colorCorrection(imagem, 1.0)
# cv2_imshow(p)


# # Define os valores de exposição, contraste e brilho (entre 0 e 2 para exposição e contraste, -50 a 50 para brilho)
# exposicao = 1.0
# contraste = 1.5
# brilho = 0.8

# Aplica o ajuste de exposição, contraste e brilho
# imagem_ajustada_2 = ajustar_exposicao(p, exposicao, contraste, brilho)

# def processo(imagem):
#     exposicao = 0.65
#     contraste = 1.5
#     brilho = 0.9
#     image = enhance_segmentation(imagem)
#     corrected = colorCorrection(image, 1.0)
#     gamma = adjust_gamma(corrected,gamma=0.85)
#     imagem_ajustada = ajustar_exposicao(gamma, exposicao, contraste, brilho)
#     return imagem_ajustada

from google.colab.patches import cv2_imshow
# # Mostra a imagem original e a imagem ajustada
# cv2_imshow(imagem)
# cv2_imshow(imagem_ajustada_2)
imagem = cv2.imread('/content/drive/MyDrive/HAZE/lanzhou_5.png')
cv2_imshow(imagem)
cv2_imshow( processo(imagem))
