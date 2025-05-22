"""
Módulo para algoritmos de compressão de imagem: Deflate e DWT.
"""
import io
import zlib
import numpy as np
from PIL import Image
import pywt
from typing import Tuple, Dict, List, Any
import config

def compress_deflate(image_array: np.ndarray, level: int = 6) -> Tuple[int, bytes]:
    """
    Comprime uma imagem usando o algoritmo Deflate (zlib).
    A imagem é primeiro convertida para PNG em memória e depois comprimida.

    Args:
        image_array (np.ndarray): Array NumPy 2D da imagem (espera-se float32, será clipado para uint8).
        level (int): Nível de compressão zlib (0-9). Padrão é 6.

    Returns:
        Tuple[int, bytes]: (tamanho_comprimido_em_bytes, stream_de_bytes_comprimidos)
    """
    # Garante que a imagem esteja no formato uint8 (0-255) para salvar como PNG
    arr_uint8 = image_array.clip(0, 255).astype(np.uint8)
    
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format='PNG')
    data_to_compress = buf.getvalue()
    
    comp_stream = zlib.compress(data_to_compress, level=level)
    compressed_size = len(comp_stream)
    
    return compressed_size, comp_stream

def compress_dwt(image_array: np.ndarray, wavelet: str = 'haar', level: int = 1, quantization_step: float = 10.0) -> Tuple[bytes, Dict[str, Any], int]:
    """
    Comprime um array de imagem 2D usando DWT, quantização e compressão zlib.

    O processo envolve:
    1. Validar entradas.
    2. Aplicar a Transformada Wavelet Discreta (DWT) multi-nível.
    3. Quantizar os coeficientes wavelet (aproximação e detalhes).
    4. Serializar os coeficientes quantizados.
    5. Comprimir o stream de bytes serializado usando zlib.
    6. Coletar metadados para descompressão.

    Args:
        image_array (np.ndarray): Array NumPy 2D da imagem (escala de cinza).
        wavelet (str): Tipo de wavelet (ex: 'haar', 'db2').
        level (int): Nível de decomposição DWT.
        quantization_step (float): Passo de quantização.

    Returns:
        Tuple[bytes, Dict[str, Any], int]:
            - compressed_stream (bytes): Bytes dos coeficientes comprimidos.
            - metadata (Dict): Dicionário com informações para descompressão.
            - compressed_size (int): Tamanho do compressed_stream em bytes.

    Raises:
        ValueError: Se as entradas forem inválidas.
    """
    # 1. Validar entradas
    if image_array.ndim != 2:
        raise ValueError("A imagem de entrada deve ser um array 2D (escala de cinza).")
    if not isinstance(level, int) or level <= 0:
        raise ValueError("O 'level' de decomposição deve ser um inteiro positivo.")
    if not isinstance(quantization_step, (float, int)) or quantization_step <= 0:
        raise ValueError("O 'quantization_step' deve ser um número positivo.")
    try:
        pywt.Wavelet(wavelet)
    except ValueError:
        raise ValueError(f"Wavelet '{wavelet}' inválida ou não reconhecida pelo PyWavelets.")

    original_dtype_name = image_array.dtype.name

    # 2. Aplicar DWT (em float64 para precisão)
    coeffs: List[Any] = pywt.wavedec2(image_array.astype(np.float64), wavelet=wavelet, level=level)

    # 3. Quantização
    quant_coeffs_structured: List[Any] = []
    coeff_shapes_structured: List[Any] = [] # Armazena os shapes dos coeficientes quantizados

    quant_coeff_dtype = config.DWT_QUANTIZED_COEFF_DTYPE
    quant_min = np.iinfo(quant_coeff_dtype).min
    quant_max = np.iinfo(quant_coeff_dtype).max

    # Coeficiente de Aproximação (cA)
    cA = coeffs[0]
    cA_quant_float = np.round(cA / quantization_step)
    cA_q = np.clip(cA_quant_float, quant_min, quant_max).astype(quant_coeff_dtype)
    quant_coeffs_structured.append(cA_q)
    coeff_shapes_structured.append(cA_q.shape)

    # Coeficientes de Detalhe (cH, cV, cD)
    for i in range(1, len(coeffs)): # coeffs[0] é cA, coeffs[1:] são os níveis de detalhe
        detail_level_coeffs_tuple = coeffs[i] # Tupla (cH_k, cV_k, cD_k)
        
        quant_detail_level_list_for_tuple: List[np.ndarray] = []
        shapes_detail_level_list_for_tuple: List[Tuple[int, ...]] = []

        for subband_coeffs in detail_level_coeffs_tuple:
            subband_quant_float = np.round(subband_coeffs / quantization_step)
            subband_q = np.clip(subband_quant_float, quant_min, quant_max).astype(quant_coeff_dtype)
            quant_detail_level_list_for_tuple.append(subband_q)
            shapes_detail_level_list_for_tuple.append(subband_q.shape)
        
        quant_coeffs_structured.append(tuple(quant_detail_level_list_for_tuple))
        coeff_shapes_structured.append(tuple(shapes_detail_level_list_for_tuple))

    # 4. Serialização dos coeficientes quantizados
    all_subbands_bytes_list: List[bytes] = []
    all_subbands_bytes_list.append(quant_coeffs_structured[0].tobytes()) # cA

    for i in range(1, len(quant_coeffs_structured)): # Níveis de detalhe
        for subband_q_in_tuple in quant_coeffs_structured[i]: # cH, cV, cD
            all_subbands_bytes_list.append(subband_q_in_tuple.tobytes())
    
    concatenated_bytes = b"".join(all_subbands_bytes_list)

    # 5. Codificação por Entropia (zlib)
    compressed_stream = zlib.compress(concatenated_bytes)
    compressed_size = len(compressed_stream)

    # 6. Coletar Metadados
    metadata = {
        'wavelet': wavelet,
        'level': level, # Nível de decomposição DWT (corresponde a len(coeffs) - 1)
        'quantization_step': quantization_step,
        'original_shape': image_array.shape,
        'original_dtype_name': original_dtype_name,
        'coeff_shapes_structured': coeff_shapes_structured,
        'quant_coeff_dtype_name': quant_coeff_dtype.__name__
    }

    return compressed_stream, metadata, compressed_size
