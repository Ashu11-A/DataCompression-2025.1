"""
Módulo para descompressão de imagens DWT.
"""
import zlib
import numpy as np
import pywt
from typing import Dict, Any, List, Tuple

def decompress_dwt(compressed_stream: bytes, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Descomprime uma imagem a partir de um stream de bytes DWT comprimido e metadados.

    Args:
        compressed_stream (bytes): Bytes dos coeficientes DWT comprimidos.
        metadata (Dict[str, Any]): Dicionário com informações para descompressão.

    Returns:
        np.ndarray: O array da imagem reconstruída.

    Raises:
        KeyError: Se alguma chave essencial estiver faltando nos metadados.
        ValueError: Se os dados desserializados não corresponderem aos shapes esperados
                    ou se o stream de bytes for insuficiente.
    """
    # 1. Extrair Metadados
    try:
        wavelet = metadata['wavelet']
        # 'level' nos metadados refere-se ao número de níveis de decomposição,
        # que é len(coeffs) - 1, ou o número de tuplas de detalhes.
        num_detail_levels = metadata['level']
        quantization_step = metadata['quantization_step']
        original_shape = metadata['original_shape']
        original_dtype = np.dtype(metadata['original_dtype_name'])
        # coeff_shapes_structured: [shape_cA, (shape_cH_L, cV_L, cD_L), ..., (shape_cH_1, cV_1, cD_1)]
        coeff_shapes_structured = metadata['coeff_shapes_structured']
        quant_coeff_dtype = np.dtype(metadata['quant_coeff_dtype_name'])
    except KeyError as e:
        raise KeyError(f"Metadado essencial faltando: {e}")

    itemsize = quant_coeff_dtype.itemsize

    # 2. Decodificação por Entropia (descompressão zlib)
    concatenated_bytes = zlib.decompress(compressed_stream)

    # 3. Desserialização e Reconstrução da Estrutura dos Coeficientes Quantizados
    quant_coeffs_reconstructed_structured: List[Any] = []
    current_pos_in_bytes = 0

    # Desserializar Aproximação (cA)
    shape_cA = coeff_shapes_structured[0]
    num_elements_cA = np.prod(shape_cA)
    num_bytes_for_cA = num_elements_cA * itemsize
    
    if current_pos_in_bytes + num_bytes_for_cA > len(concatenated_bytes):
        raise ValueError("Stream de bytes insuficiente para reconstruir cA.")
        
    cA_q_bytes = concatenated_bytes[current_pos_in_bytes : current_pos_in_bytes + num_bytes_for_cA]
    cA_q = np.frombuffer(cA_q_bytes, dtype=quant_coeff_dtype).reshape(shape_cA)
    quant_coeffs_reconstructed_structured.append(cA_q)
    current_pos_in_bytes += num_bytes_for_cA

    # Desserializar Detalhes (iterar pelos 'num_detail_levels')
    for i in range(num_detail_levels):
        # O i-ésimo nível de detalhe corresponde ao índice (i+1) em coeff_shapes_structured
        if (i + 1) >= len(coeff_shapes_structured):
            raise ValueError(f"Estrutura de shapes de coeficientes incompleta para o nível de detalhe {i+1}.")

        shapes_in_detail_tuple: Tuple[Tuple[int, ...], ...] = coeff_shapes_structured[i + 1]
        if not isinstance(shapes_in_detail_tuple, tuple):
            raise ValueError(f"Shapes para o nível de detalhe {i+1} não é uma tupla.")

        detail_tuple_reconstructed_list: List[np.ndarray] = []
        for shape_detail_subband in shapes_in_detail_tuple: # (cH, cV, cD)
            num_elements_detail = np.prod(shape_detail_subband)
            num_bytes_for_detail_subband = num_elements_detail * itemsize

            if current_pos_in_bytes + num_bytes_for_detail_subband > len(concatenated_bytes):
                raise ValueError(f"Stream de bytes insuficiente para sub-banda de detalhe {shape_detail_subband}.")

            detail_bytes = concatenated_bytes[current_pos_in_bytes : current_pos_in_bytes + num_bytes_for_detail_subband]
            detail_q_subband = np.frombuffer(detail_bytes, dtype=quant_coeff_dtype).reshape(shape_detail_subband)
            detail_tuple_reconstructed_list.append(detail_q_subband)
            current_pos_in_bytes += num_bytes_for_detail_subband
        quant_coeffs_reconstructed_structured.append(tuple(detail_tuple_reconstructed_list))

    if current_pos_in_bytes != len(concatenated_bytes):
        # Isso pode acontecer se houver padding ou erro na serialização/desserialização.
        # Considerar se isso deve ser um erro ou um aviso.
        print(f"Atenção: {len(concatenated_bytes) - current_pos_in_bytes} bytes "
              f"não lidos ou bytes faltando no stream após a desserialização.")

    # 4. Dequantização
    dequant_coeffs: List[Any] = []
    # Aproximação (converter para float64 para IDWT)
    dequant_coeffs.append(quant_coeffs_reconstructed_structured[0].astype(np.float64) * quantization_step)

    # Detalhes
    for i in range(1, len(quant_coeffs_reconstructed_structured)): # Tuplas de detalhes
        dequant_detail_level_tuple_list: List[np.ndarray] = []
        for subband_q_in_tuple in quant_coeffs_reconstructed_structured[i]:
            dequant_detail_level_tuple_list.append(subband_q_in_tuple.astype(np.float64) * quantization_step)
        dequant_coeffs.append(tuple(dequant_detail_level_tuple_list))

    # 5. Aplicar Transformada Wavelet Inversa (IDWT)
    reconstructed_array = pywt.waverec2(dequant_coeffs, wavelet=wavelet)

    # 6. Pós-processamento
    h_orig, w_orig = original_shape[:2]
    # IDWT pode resultar em dimensões ligeiramente diferentes. Cortar para o original.
    reconstructed_array = reconstructed_array[:h_orig, :w_orig]

    # Clipar e converter para o tipo de dados original
    if np.issubdtype(original_dtype, np.integer):
        type_info = np.iinfo(original_dtype)
        reconstructed_array = np.clip(reconstructed_array, type_info.min, type_info.max)
    # Para float, a conversão de tipo abaixo geralmente é suficiente.
    
    return reconstructed_array.astype(original_dtype)
