"""
Módulo para carregamento e pré-processamento de imagens.
"""
import os
import numpy as np
from PIL import Image
import rawpy
import config

def load_image(path: str) -> np.ndarray:
    """
    Carrega uma imagem de um arquivo, convertendo-a para escala de cinza e tipo float32.
    Suporta formatos de imagem padrão e arquivos RAW (.nef).

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
            # Documentação sobre os parâmetros de postprocess:
            # use_camera_wb=True: Aplica o balanço de branco conforme definido pela câmera.
            # no_auto_bright=True: Evita ajustes automáticos de brilho.
            # output_bps=8: Gera uma imagem RGB com 8 bits por canal.
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            
            # Conversão para escala de cinza usando coeficientes de luminância padrão.
            # rgb[..., :3] seleciona os canais R, G, B.
            gray = np.dot(rgb[..., :3], config.RGB_TO_GRAY_COEFFS)
            return gray.astype(np.float32)
        else:
            img = Image.open(path).convert('L') # 'L' para converter para escala de cinza
            return np.array(img, dtype=np.float32)
    except Exception as e:
        raise Exception(f"Erro ao carregar ou processar a imagem {path}: {e}")

