"""
Módulo com funções utilitárias para manipulação de arquivos CSV.
"""
import csv
from typing import List, Dict, Any
import config # Para chaves de dicionário e formatação

def _format_csv_value(key: str, value: Any) -> str:
    """Formata um valor para escrita no CSV."""
    if isinstance(value, float):
        if value == float('inf'):
            return 'inf'
        if key == config.KEY_SSIM:
            return f"{value:.4f}"
        if key in [config.KEY_BPP, config.KEY_COMPRESSION_TIME_S, config.KEY_COMPRESSION_RATIO]:
            return f"{value:.4f}" # Aumenta precisão para taxa também
        return f"{value:.2f}" # Padrão para outros floats
    if value is None: # Para wavelet_name em deflate, por exemplo
        return ''
    return str(value)

def initialize_csv(csv_path: str, fieldnames: List[str]) -> None:
    """
    Cria um arquivo CSV e escreve o cabeçalho.

    Args:
        csv_path (str): Caminho para o arquivo CSV.
        fieldnames (List[str]): Lista dos nomes das colunas.
    """
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
    except IOError as e:
        print(f"Erro ao inicializar CSV em {csv_path}: {e}")
        raise # Re-levanta a exceção pois é crítico

def append_result_to_csv(result_dict: Dict[str, Any], csv_path: str, fieldnames: List[str]) -> None:
    """
    Adiciona uma linha de resultado formatada a um arquivo CSV existente.

    Args:
        result_dict (Dict[str, Any]): Dicionário contendo os dados da linha.
        csv_path (str): Caminho para o arquivo CSV.
        fieldnames (List[str]): Lista dos nomes das colunas (para DictWriter).
    """
    formatted_row = {key: _format_csv_value(key, result_dict.get(key)) for key in fieldnames}
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(formatted_row)
    except IOError as e:
        print(f"Erro ao adicionar linha ao CSV {csv_path} para dados {result_dict.get(config.KEY_PARAMETERS)}: {e}")

def _get_sort_key_for_results(result_dict: Dict[str, Any]) -> tuple:
    """
    Chave de ordenação para os resultados: prioriza maior taxa de compressão,
    depois menor tamanho comprimido.
    """
    ratio_str = result_dict.get(config.KEY_COMPRESSION_RATIO, '0.0')
    comp_size_str = result_dict.get(config.KEY_COMPRESSED_SIZE_BYTES, str(float('inf')))

    try:
        # Tenta converter para float, tratando 'inf'
        ratio_val = float(ratio_str) if ratio_str != 'inf' else float('inf')
        comp_size_val = float(comp_size_str) if comp_size_str != 'inf' else float('inf')
    except ValueError:
        # Fallback se a conversão falhar (improvável com _format_csv_value)
        ratio_val = 0.0
        comp_size_val = float('inf')
        print(f"Aviso: Erro ao converter valores para ordenação: {ratio_str}, {comp_size_str}")

    # Queremos ordenar por taxa de compressão decrescente (-ratio_val)
    # e tamanho comprimido crescente (comp_size_val)
    return (-ratio_val, comp_size_val)

def sort_and_rewrite_csv(
    collected_results: List[Dict[str, Any]], 
    csv_path: str, 
    fieldnames: List[str]
) -> None:
    """
    Ordena os resultados coletados e reescreve o arquivo CSV.

    Args:
        collected_results (List[Dict[str, Any]]): Lista de dicionários de resultados (não formatados).
        csv_path (str): Caminho para o arquivo CSV a ser reescrito.
        fieldnames (List[str]): Nomes das colunas.
    """
    # Os resultados em collected_results estão com tipos numéricos (float, int)
    # A chave de ordenação deve usar esses valores numéricos diretamente.
    def sort_key_numeric(res_dict_numeric: Dict[str, Any]) -> tuple:
        ratio = res_dict_numeric.get(config.KEY_COMPRESSION_RATIO, 0.0)
        # Se ratio for 'inf', queremos que seja o maior
        ratio_val = float('inf') if ratio == float('inf') else float(ratio)
        
        comp_size = res_dict_numeric.get(config.KEY_COMPRESSED_SIZE_BYTES, float('inf'))
        comp_size_val = float(comp_size) # Já deve ser numérico ou inf

        return (-ratio_val, comp_size_val)

    sorted_results_numeric = sorted(collected_results, key=sort_key_numeric)
    
    # Reabre o CSV em modo 'w' para sobrescrever com os dados ordenados e formatados
    initialize_csv(csv_path, fieldnames) # Escreve o cabeçalho novamente
    for res_numeric in sorted_results_numeric:
        append_result_to_csv(res_numeric, csv_path, fieldnames) # Adiciona formatado
