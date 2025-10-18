"""
Módulo para carregamento e pré-processamento de imagens.
"""
import os
import numpy as np
from PIL import Image
import rawpy
from src.config import RGB_TO_GRAY_COEFFS

def load_image(path: str) -> np.ndarray:
    """
    Carrega uma imagem de um arquivo, convertendo-a para escala de cinza e tipo float32.
    Para arquivos .nef, salva uma versão .png comprimida antes de retornar.

    Args:
        path (str): Caminho para o arquivo de imagem.

    Returns:
        np.ndarray: Array NumPy 2D representando a imagem em escala de cinza (float32).

    Raises:
        FileNotFoundError: Se o arquivo de imagem não for encontrado.
        Exception: Para outros erros de carregamento ou processamento.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado em: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.nef':
            # Carrega e processa a imagem RAW.
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            
            # Converte para escala de cinza.
            gray_float = np.dot(rgb[..., :3], RGB_TO_GRAY_COEFFS)

            # --- Início da Modificação: Compressão para PNG ---
            
            # Converte para uint8 para salvar como imagem.
            gray_uint8 = gray_float.astype(np.uint8)
            img_pil = Image.fromarray(gray_uint8, mode='L')

            # Gera o caminho do arquivo de saída trocando a extensão.
            output_path = os.path.splitext(path)[0] + '.png'
            
            # Salva a imagem em PNG (que usa DEFLATE) com nível de compressão 9.
            img_pil.save(output_path, 'PNG', compress_level=9)
            
            # --- Fim da Modificação ---

            return gray_float.astype(np.float32)
        else:
            img = Image.open(path).convert('L')
            return np.array(img, dtype=np.float32)
    except Exception as e:
        raise Exception(f"Erro ao carregar ou processar a imagem {path}: {e}")
      
load_image("photo.nef")