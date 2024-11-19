from skimage import color, io
import numpy as np
import cv2

def getNIQE(img):# Carregue a imagem usando scikit-image (substitua 'imagem.jpg' pelo caminho da sua imagem)
    # Converta a imagem para escala de cinza
    imagem_gray = color.rgb2gray(img)

    # Calcule a média e o desvio padrão do contraste local
    mean_local_contrast = np.mean(np.abs(np.gradient(imagem_gray)))
    std_local_contrast = np.std(np.abs(np.gradient(imagem_gray)))

    # Calcule o NIQE
    niqe_score = 1.0 / (1 + 6.6 * mean_local_contrast + 0.228 * std_local_contrast)

    return niqe_score

def calculate_mscn_coefficients(image):
    c = np.fft.fft2(image)
    c_shifted = np.fft.fftshift(c)
    magnitude = np.abs(c_shifted)
    log_magnitude = np.log(1.0 + magnitude)
    c_shifted_real = np.real(c_shifted)
    c_shifted_imag = np.imag(c_shifted)
    return c_shifted_real, c_shifted_imag, magnitude, log_magnitude

def calculate_mscn_features(image):
    c_shifted_real, c_shifted_imag, _, log_magnitude = calculate_mscn_coefficients(image)
    std_dev = np.std(log_magnitude)
    mean = np.mean(log_magnitude)
    return [std_dev, mean]

def brisque_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ms_std_dev, ms_mean = calculate_mscn_features(gray)
    return [ms_std_dev, ms_mean]

def getBRISQUE(image):
    features = brisque_features(image)
    weights = [-0.0977446, 0.0270277, 0.00090095, 0.0793246, 0.0476165, -0.033992, -0.0535509, 0.276186, 0.189205, 0.255546,
               0.120626, 0.0471861, -0.18469, 0.154051, -0.173411, -0.413456]
    intercept = 18.9217
    score = intercept
    for i in range(len(features)):
        score += features[i] * weights[i]
    return score

# Exemplo de uso
image_path = '/content/drive/MyDrive/boa.jpg'
image = cv2.imread(image_path)
score1 = getBRISQUE(image)
score2 = getNIQE(image)
print('Pontuação BRISQUE da imagem:', score1)
print('Pontuação NIQE da imagem:', score2)
