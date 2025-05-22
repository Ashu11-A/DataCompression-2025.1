"""
Módulo para gerenciar e executar tarefas de compressão,
incluindo processamento paralelo e coleta de resultados.
"""
import os
import time
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pywt # Para dwt_max_level
from typing import Dict, Any, List, Tuple

import config # Constantes globais
from core import compression, decompression, metrics # Funções de core
from plotting import individual_plots, summary_plots # Funções de plotagem
from utils import file_utils, csv_utils # Funções utilitárias

def _prepare_task_result_dict(original_pixel_bytes: int, img_h: int, img_w: int, params: Dict) -> Dict[str, Any]:
    """Prepara um dicionário base para os resultados da tarefa."""
    return {
        config.KEY_ORIGINAL_SIZE_BYTES: original_pixel_bytes,
        config.KEY_DIMENSIONS: f'{img_h}x{img_w}',
        config.KEY_RAW_PARAMS: params.copy() # Salva uma cópia dos parâmetros crus
    }

def _finalize_task_result_dict(
    result_dict: Dict[str, Any], 
    compressed_size: int, 
    proc_time: float, 
    num_pixels: int
) -> None:
    """Adiciona métricas calculadas ao dicionário de resultados."""
    result_dict[config.KEY_COMPRESSED_SIZE_BYTES] = compressed_size
    result_dict[config.KEY_COMPRESSION_TIME_S] = proc_time
    
    original_size = result_dict[config.KEY_ORIGINAL_SIZE_BYTES]
    if compressed_size > 0:
        result_dict[config.KEY_COMPRESSION_RATIO] = original_size / compressed_size
        result_dict[config.KEY_BPP] = (compressed_size * 8) / num_pixels
    else:
        result_dict[config.KEY_COMPRESSION_RATIO] = float('inf')
        result_dict[config.KEY_BPP] = float('inf')


def process_compression_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processa uma única tarefa de compressão (Deflate ou DWT).

    Args:
        task_args (Dict[str, Any]): Dicionário contendo todos os argumentos
                                     necessários para a tarefa.
                                     Campos esperados: 'algo_type', 'base_original_arr',
                                     'params', 'original_pixel_bytes', 'img_h', 'img_w',
                                     'num_pixels', 'images_output_dir', 'plots_output_dir',
                                     'streams_output_dir', 'save_streams_flag'.
    Returns:
        Dict[str, Any]: Dicionário com os resultados da compressão.
    """
    algo_type = task_args['algo_type']
    base_original_arr = task_args['base_original_arr']
    params = task_args['params'] # Ex: {'level': 6} ou {'wavelet': 'haar', 'level': 1, 'quant': 10}
    
    original_pixel_bytes = task_args['original_pixel_bytes']
    img_h, img_w = task_args['img_h'], task_args['img_w']
    num_pixels = task_args['num_pixels']
    
    images_output_dir = task_args['images_output_dir'] # Para save_reconstructed_image
    plots_output_dir = task_args['plots_output_dir']   # Para plot_image_comparison
    streams_output_dir = task_args['streams_output_dir']
    save_streams_flag = task_args['save_streams_flag']

    result_dict = _prepare_task_result_dict(original_pixel_bytes, img_h, img_w, params)
    
    compressed_size = 0
    proc_time = 0.0
    
    if algo_type == 'deflate':
        level = params['level']
        result_dict[config.KEY_ALGORITHM] = 'Deflate'
        result_dict[config.KEY_PARAMETERS] = f'level={level}'
        result_dict[config.KEY_WAVELET_NAME] = '' # Deflate não usa wavelet
    
        start_time = time.time()
        compressed_size, compressed_stream = compression.compress_deflate(base_original_arr, level=level)
        proc_time = time.time() - start_time
        
        if save_streams_flag:
            stream_filename = f"image_deflate_level{level}.zlib"
            file_utils.save_stream(compressed_stream, streams_output_dir, stream_filename)
        
        # Métricas para compressão sem perdas
        result_dict[config.KEY_PSNR] = float('inf')
        result_dict[config.KEY_SSIM] = 1.0

    elif algo_type == 'dwt':
        wavelet, dwt_level, quant = params['wavelet'], params['level'], params['quant']
        result_dict[config.KEY_ALGORITHM] = 'DWT'
        result_dict[config.KEY_WAVELET_NAME] = wavelet # Usado para agrupar resultados DWT
        result_dict[config.KEY_PARAMETERS] = f"wavelet={wavelet},level={dwt_level},quant={quant}"
        
        start_time = time.time()
        try:
            compressed_stream, metadata, compressed_size = compression.compress_dwt(
                base_original_arr, wavelet=wavelet, level=dwt_level, quantization_step=quant
            )
            reconstructed_image = decompression.decompress_dwt(compressed_stream, metadata)
        except Exception as e:
            print(f"Erro durante compressão/descompressão DWT para {params}: {e}")
            # Preencher com valores de erro ou pular
            result_dict[config.KEY_PSNR] = np.nan 
            result_dict[config.KEY_SSIM] = np.nan
            compressed_size = -1 # Indicar erro
            reconstructed_image = np.zeros_like(base_original_arr) # Placeholder
        proc_time = time.time() - start_time
        
        if save_streams_flag and compressed_size != -1:
            stream_filename = f"image_dwt_{wavelet}_level{dwt_level}_quant{quant}.dwtz"
            file_utils.save_stream(compressed_stream, streams_output_dir, stream_filename)
        
        if compressed_size != -1:
            # Salvar imagem reconstruída e plot de comparação
            fn_base = f"dwt_{wavelet}_level{dwt_level}_quant{quant}"
            individual_plots.save_reconstructed_image(reconstructed_image, images_output_dir, fn_base)
            individual_plots.plot_image_comparison(
                base_original_arr, reconstructed_image, plots_output_dir, fn_base,
                reconstructed_title=f'Reconstruída DWT ({wavelet}, L{dwt_level}, Q{quant})'
            )
            
            result_dict[config.KEY_PSNR] = metrics.calculate_psnr(base_original_arr, reconstructed_image)
            result_dict[config.KEY_SSIM] = metrics.calculate_ssim(base_original_arr, reconstructed_image)
    else:
        raise ValueError(f"Tipo de algoritmo desconhecido: {algo_type}")

    _finalize_task_result_dict(result_dict, compressed_size, proc_time, num_pixels)
    return result_dict


def run_parameter_tests(
    base_original_arr: np.ndarray,
    output_dir_base_timestamp: str, 
    save_streams_flag: bool, 
    num_workers: int = 4
) -> str:
    """
    Executa uma série de testes de compressão com diferentes parâmetros em paralelo.

    Args:
        base_original_arr (np.ndarray): Array da imagem original carregada.
        output_dir_base_timestamp (str): Diretório base com timestamp para salvar todos os resultados.
        save_streams_flag (bool): Flag para salvar os streams de bytes comprimidos.
        num_workers (int): Número de workers para o ProcessPoolExecutor.

    Returns:
        str: Caminho para o arquivo CSV final com os resultados ordenados.
    """
    # Criação de subdiretórios específicos para esta execução de teste
    images_output_dir = os.path.join(output_dir_base_timestamp, config.DIR_IMAGES)
    summary_plots_output_dir = os.path.join(output_dir_base_timestamp, config.DIR_SUMMARY_PLOTS)
    individual_plots_output_dir = os.path.join(output_dir_base_timestamp, config.DIR_INDIVIDUAL_PLOTS)
    streams_output_dir = os.path.join(output_dir_base_timestamp, config.DIR_STREAMS)

    file_utils.ensure_dir_exists(images_output_dir)
    file_utils.ensure_dir_exists(summary_plots_output_dir)
    file_utils.ensure_dir_exists(individual_plots_output_dir)
    if save_streams_flag:
        file_utils.ensure_dir_exists(streams_output_dir)
        # Salva o stream da imagem original (não comprimida, em float32)
        file_utils.save_stream(base_original_arr.tobytes(), streams_output_dir, config.ORIGINAL_IMAGE_STREAM_FILENAME)

    original_pixel_bytes = base_original_arr.nbytes
    img_h, img_w = base_original_arr.shape
    num_pixels = img_h * img_w
    print(f"Imagem base para testes: {img_w}x{img_h} pixels, {original_pixel_bytes} bytes (dtype: {base_original_arr.dtype})")

    task_args_template = {
        'base_original_arr': base_original_arr,
        'original_pixel_bytes': original_pixel_bytes,
        'img_h': img_h, 'img_w': img_w, 'num_pixels': num_pixels,
        'images_output_dir': images_output_dir,       # Para salvar .png reconstruído
        'plots_output_dir': individual_plots_output_dir, # Para plots de comparação individual
        'streams_output_dir': streams_output_dir,
        'save_streams_flag': save_streams_flag
    }
    
    tasks_to_process: List[Dict[str, Any]] = []
    # Contagem de tarefas esperadas por grupo para saber quando plotar sumários
    expected_counts_by_group: Dict[str, int] = {} 
    # Acumulador de resultados por grupo para plotagem de sumário
    results_by_group: Dict[str, List[Dict[str, Any]]] = {} 

    # 1. Definir Tarefas Deflate
    group_key_deflate = 'deflate_summary_group' # Chave única para o grupo Deflate
    expected_counts_by_group[group_key_deflate] = 0
    results_by_group[group_key_deflate] = []
    for level in config.DEFAULT_DEFLATE_LEVELS:
        tasks_to_process.append({**task_args_template, 'algo_type': 'deflate', 'params': {'level': level}})
        expected_counts_by_group[group_key_deflate] += 1

    # 2. Definir Tarefas DWT
    for wavelet_name in config.DEFAULT_DWT_WAVELETS:
        # Cada wavelet terá seu próprio grupo para sumário
        group_key_dwt_wavelet = f'dwt_summary_group_{wavelet_name}'
        expected_counts_by_group[group_key_dwt_wavelet] = 0
        results_by_group[group_key_dwt_wavelet] = []

        for dwt_level, quant_step in itertools.product(
            config.DEFAULT_DWT_LEVELS,
            config.DEFAULT_DWT_QUANT_STEPS
        ):
            try:
                # Verifica se o nível de decomposição é válido para a imagem e wavelet
                max_level_allowed = pywt.dwt_max_level(min(img_h, img_w), pywt.Wavelet(wavelet_name).dec_len)
                if dwt_level <= max_level_allowed:
                    tasks_to_process.append({
                        **task_args_template, 
                        'algo_type': 'dwt', 
                        'params': {'wavelet': wavelet_name, 'level': dwt_level, 'quant': quant_step}
                    })
                    expected_counts_by_group[group_key_dwt_wavelet] +=1
                else:
                    print(f"Aviso: Nível DWT {dwt_level} para wavelet '{wavelet_name}' excede o máximo permitido ({max_level_allowed}). Pulando.")
            except ValueError as e: # Caso a wavelet não seja válida (embora já deva ser verificada em compress_dwt)
                print(f"Aviso: Wavelet '{wavelet_name}' inválida ao definir tarefas: {e}. Pulando.")
                break # Pula para a próxima wavelet se esta for inválida


    csv_path = os.path.join(output_dir_base_timestamp, config.CSV_RESULTS_FILENAME)
    # Abre o CSV para escrita do cabeçalho
    csv_utils.initialize_csv(csv_path, config.CSV_FIELDNAMES)
    
    collected_results_for_final_sort: List[Dict[str, Any]] = [] 

    print(f"\nIniciando {len(tasks_to_process)} tarefas de compressão com {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Mapeia futuros para uma string de identificação da tarefa para logging
        future_to_task_id = {
            executor.submit(process_compression_task, task_args): 
            f"{task_args['algo_type']}_{task_args['params']}" 
            for task_args in tasks_to_process
        }
        
        for i, future in enumerate(as_completed(future_to_task_id)):
            task_id_str = future_to_task_id[future]
            try:
                # result_dict_numeric contém os valores numéricos como estão
                result_dict_numeric = future.result() 
                if result_dict_numeric.get(config.KEY_COMPRESSED_SIZE_BYTES, -1) == -1:
                    print(f"({i+1}/{len(tasks_to_process)}) Tarefa {task_id_str} falhou ou foi pulada. Verifique logs anteriores.")
                    continue # Pula para o próximo futuro se a tarefa falhou

                collected_results_for_final_sort.append(result_dict_numeric)

                # Salva a linha formatada no CSV imediatamente
                csv_utils.append_result_to_csv(result_dict_numeric, csv_path, config.CSV_FIELDNAMES)
                print(f"({i+1}/{len(tasks_to_process)}) Resultado para: {task_id_str} salvo no CSV.")

                # Agrupa resultados e plota sumários quando um grupo estiver completo
                algo = result_dict_numeric[config.KEY_ALGORITHM]
                current_task_group_key = None
                if algo == 'Deflate':
                    current_task_group_key = group_key_deflate
                elif algo == 'DWT':
                    wavelet_n = result_dict_numeric.get(config.KEY_WAVELET_NAME)
                    if wavelet_n:
                        current_task_group_key = f'dwt_summary_group_{wavelet_n}'
                
                if current_task_group_key and current_task_group_key in results_by_group:
                    results_by_group[current_task_group_key].append(result_dict_numeric)
                    
                    # Verifica se todos os resultados esperados para este grupo foram coletados
                    if len(results_by_group[current_task_group_key]) == expected_counts_by_group.get(current_task_group_key, -1):
                        print(f"Todos os testes para o grupo '{current_task_group_key}' concluídos. Gerando gráfico de sumário...")
                        if current_task_group_key == group_key_deflate:
                            summary_plots.plot_deflate_summary_chart(
                                results_by_group[current_task_group_key], 
                                summary_plots_output_dir
                            )
                        elif current_task_group_key.startswith('dwt_summary_group_'):
                            # Extrai o nome da wavelet da chave do grupo
                            actual_wavelet_name = current_task_group_key.split('dwt_summary_group_')[1]
                            summary_plots.plot_dwt_wavelet_summary_chart(
                                actual_wavelet_name, 
                                results_by_group[current_task_group_key], 
                                summary_plots_output_dir
                            )
            except Exception as exc:
                print(f"Tarefa {task_id_str} gerou uma exceção durante a coleta do resultado: {exc}")

    print(f"\nTodas as {len(collected_results_for_final_sort)} tarefas válidas concluídas.")
    
    # Reordena o CSV final com base na taxa de compressão e tamanho
    if collected_results_for_final_sort:
        print(f"Reordenando o CSV final em: {csv_path}")
        csv_utils.sort_and_rewrite_csv(collected_results_for_final_sort, csv_path, config.CSV_FIELDNAMES)
    else:
        print("Nenhum resultado válido coletado para ordenação final do CSV.")
            
    print(f"\nTestes em lote concluídos. Resultados finais salvos e ordenados em {csv_path}")
    return csv_path
