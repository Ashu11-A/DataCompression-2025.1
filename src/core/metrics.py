"""
Módulo para cálculo de métricas de qualidade de imagem.
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage # Renomeado para evitar conflito
import config

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel_value: float = config.MAX_PIXEL_VALUE_8BIT) -> float:
    """
    Calcula o Peak Signal-to-Noise Ratio (PSNR) entre duas imagens.

    Args:
        original (np.ndarray): Imagem original.
        reconstructed (np.ndarray): Imagem reconstruída/comprimida.
        max_pixel_value (float): Valor máximo possível para um pixel (padrão 255.0).

    Returns:
        float: Valor do PSNR em dB. Retorna float('inf') se as imagens forem idênticas.
               Retorna np.nan se houver erro de dimensão.
    """
    if original.shape != reconstructed.shape:
        print("Aviso PSNR: Dimensões das imagens original e reconstruída não coincidem.")
        return np.nan # Ou levantar um erro

    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        return float('inf') # Imagens idênticas
    
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray, data_range: float = config.MAX_PIXEL_VALUE_8BIT) -> float:
    """
    Calcula o Structural Similarity Index (SSIM) entre duas imagens.

    Args:
        original (np.ndarray): Imagem original.
        reconstructed (np.ndarray): Imagem reconstruída/comprimida.
        data_range (float): A variação dos dados da imagem (max_val - min_val).
                            Padrão é 255.0 para imagens de 8 bits.

    Returns:
        float: Valor do SSIM. Retorna np.nan se houver erro de dimensão.
    """
    if original.shape != reconstructed.shape:
        print("Aviso SSIM: Dimensões das imagens original e reconstruída não coincidem.")
        return np.nan # Ou levantar um erro

    # SSIM espera float, mas o tipo exato pode depender da versão do skimage.
    # A conversão para float32 é geralmente segura e comum.
    # A função ssim_skimage lida com a normalização se data_range for fornecido.
    return ssim_skimage(original.astype(np.float32), 
                        reconstructed.astype(np.float32), 
                        data_range=data_range)

