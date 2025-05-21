#!/usr/bin/env python3
"""
Script para compressão de imagens (incluindo RAW .nef) utilizando dois métodos:
1. Deflate (zlib) - compressão sem perdas (aplicado a uma representação PNG da imagem)
2. Discrete Wavelet Transform (DWT) - compressão com perdas via quantização de coeficientes
Executa testes em paralelo, salva resultados individuais progressivamente no CSV,
e ao final reordena o CSV por poder de compressão e tamanho.
"""
import argparse
import zlib
import os
import io
import numpy as np
import matplotlib
# try: # Descomente se tiver problemas de backend com matplotlib em paralelo
#     matplotlib.use('Agg')
# except ImportError:
#     print("Atenção: backend 'Agg' para matplotlib não pôde ser definido.")
import matplotlib.pyplot as plt
import itertools
import csv
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Bibliotecas para processamento de imagem
from PIL import Image
import pywt
import rawpy
from skimage.metrics import structural_similarity


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.nef':
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            gray = np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])
            return gray.astype(np.float32)
    else:
        img = Image.open(path).convert('L')
        return np.array(img, dtype=np.float32)


def compress_deflate(image_array, level=6):
    arr_uint8 = image_array.clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format='PNG')
    data_to_compress = buf.getvalue()
    comp_stream = zlib.compress(data_to_compress, level)
    compressed_size = len(comp_stream)
    return compressed_size, comp_stream


def compress_dwt(image_array, wavelet='haar', level=1, quantization=10):
    coeffs = pywt.wavedec2(image_array, wavelet=wavelet, level=level)
    quant_coeffs = []
    for coeff_group in coeffs:
        if isinstance(coeff_group, tuple):
            quant_coeffs.append(tuple(
                np.round(subband / quantization).astype(np.int16)
                for subband in coeff_group
            ))
        else:
            quant_coeffs.append(np.round(coeff_group / quantization).astype(np.int16))

    buf = io.BytesIO()
    for coeff_group in quant_coeffs:
        if isinstance(coeff_group, tuple):
            for subband in coeff_group:
                buf.write(subband.tobytes())
        else:
            buf.write(coeff_group.tobytes())
    quantized_coeffs_bytes = buf.getvalue()
    comp_stream = zlib.compress(quantized_coeffs_bytes)
    compressed_size = len(comp_stream)

    dequant_coeffs = []
    for coeff_group in quant_coeffs:
        if isinstance(coeff_group, tuple):
            dequant_coeffs.append(tuple(
                subband.astype(np.float32) * quantization for subband in coeff_group
            ))
        else:
            dequant_coeffs.append(coeff_group.astype(np.float32) * quantization)

    reconstructed = pywt.waverec2(dequant_coeffs, wavelet=wavelet)
    reconstructed = reconstructed[:image_array.shape[0], :image_array.shape[1]]
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed, compressed_size, comp_stream


def save_images_for_task(original_arr, reconstructed_arr, output_dir, filename_base):
    os.makedirs(output_dir, exist_ok=True)
    original_uint8 = original_arr.clip(0, 255).astype(np.uint8)
    reconstructed_uint8 = reconstructed_arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(original_uint8).save(os.path.join(output_dir, f"{filename_base}_original.png"))
    Image.fromarray(reconstructed_uint8).save(os.path.join(output_dir, f"{filename_base}_reconstructed.png"))


def plot_images_for_task(original_arr, reconstructed_arr, output_dir, filename_base):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_arr, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Imagem Original'); axes[0].axis('off')
    axes[1].imshow(reconstructed_arr, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Imagem Reconstruída (DWT)'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_base}_comparison.png"))
    plt.close(fig)

def calculate_psnr_metric(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def process_compression_task(task_args):
    # Extração de argumentos
    algo_type = task_args['algo_type']
    base_original_arr = task_args['base_original_arr']
    params = task_args['params']
    original_pixel_bytes = task_args['original_pixel_bytes']
    img_h, img_w = task_args['img_h'], task_args['img_w']
    num_pixels = task_args['num_pixels']
    images_dir = task_args['images_dir']
    plots_dir = task_args['plots_dir']
    streams_dir = task_args['streams_dir']
    save_streams_flag = task_args['save_streams_flag']

    result_dict = {
        'Tamanho Original Pixels (bytes)': original_pixel_bytes,
        'Dimensões (HxW)': f'{img_h}x{img_w}',
    }
    comp_size = 0
    proc_time = 0.0

    if algo_type == 'deflate':
        level = params['level']
        result_dict['Algoritmo'] = 'Deflate'
        result_dict['Parâmetros'] = f'level={level}'
        start_time = time.time()
        comp_size, comp_stream = compress_deflate(base_original_arr, level=level)
        proc_time = time.time() - start_time
        if save_streams_flag:
            with open(os.path.join(streams_dir, f"image_deflate_level{level}.zlib"), 'wb') as f: f.write(comp_stream)
        result_dict['PSNR (dB)'] = float('inf')
        result_dict['SSIM'] = 1.0
    elif algo_type == 'dwt':
        wavelet, dwt_level, quant = params['wavelet'], params['level'], params['quant']
        result_dict['Algoritmo'] = 'DWT'
        result_dict['Parâmetros'] = f"wavelet={wavelet},level={dwt_level},quant={quant}"
        start_time = time.time()
        recon_img, comp_size, comp_stream = compress_dwt(base_original_arr, wavelet=wavelet, level=dwt_level, quantization=quant)
        proc_time = time.time() - start_time
        if save_streams_flag:
            with open(os.path.join(streams_dir, f"image_dwt_{wavelet}_level{dwt_level}_quant{quant}.dwtz"), 'wb') as f: f.write(comp_stream)
        fn_base = f"dwt_{wavelet}_level{dwt_level}_quant{quant}"
        save_images_for_task(base_original_arr, recon_img, images_dir, fn_base)
        plot_images_for_task(base_original_arr, recon_img, plots_dir, fn_base)
        result_dict['PSNR (dB)'] = calculate_psnr_metric(base_original_arr, recon_img)
        result_dict['SSIM'] = structural_similarity(base_original_arr.astype(np.float32), recon_img.astype(np.float32), data_range=255.0)
    else:
        raise ValueError(f"Tipo de algoritmo desconhecido: {algo_type}")

    result_dict['Tamanho Comprimido (bytes)'] = comp_size
    result_dict['Tempo de Compressão (s)'] = proc_time
    result_dict['Taxa de Compressão (Original/Comprimido)'] = original_pixel_bytes / comp_size if comp_size > 0 else float('inf')
    result_dict['Bits Per Pixel (bpp)'] = (comp_size * 8) / num_pixels if num_pixels > 0 else float('inf')
    return result_dict

def run_parameter_tests(input_path, output_dir_base, save_streams_flag, num_workers=4):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, f"comp_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = os.path.join(output_dir, "images"); os.makedirs(images_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    streams_dir = os.path.join(output_dir, "streams")
    if save_streams_flag: os.makedirs(streams_dir, exist_ok=True)

    print(f"Carregando imagem de base: {input_path}")
    base_original_arr = load_image(input_path)
    original_pixel_bytes = base_original_arr.nbytes
    img_h, img_w = base_original_arr.shape; num_pixels = img_h * img_w
    print(f"Imagem base carregada: {img_w}x{img_h} pixels, {original_pixel_bytes} bytes (float32)")

    task_args_template = {'base_original_arr': base_original_arr,'original_pixel_bytes': original_pixel_bytes,
                          'img_h': img_h, 'img_w': img_w, 'num_pixels': num_pixels,
                          'images_dir': images_dir, 'plots_dir': plots_dir, 'streams_dir': streams_dir,
                          'save_streams_flag': save_streams_flag}
    tasks_to_process = []
    for level in [1, 3, 6, 9]: # Deflate levels
        tasks_to_process.append({**task_args_template, 'algo_type': 'deflate', 'params': {'level': level}})
    for wavelet, dwt_level, quant in itertools.product(['haar', 'db1', 'db4', 'sym2', 'coif1'], [1, 2, 3, 4], [1, 5, 10, 20, 40]): # DWT params
        if dwt_level <= pywt.dwt_max_level(min(img_h, img_w), pywt.Wavelet(wavelet).dec_len):
            tasks_to_process.append({**task_args_template, 'algo_type': 'dwt', 
                                     'params': {'wavelet': wavelet, 'level': dwt_level, 'quant': quant}})

    csv_path = os.path.join(output_dir, "compression_results_sorted.csv")
    fieldnames = ['Algoritmo', 'Parâmetros', 'Tamanho Original Pixels (bytes)', 'Dimensões (HxW)',
                  'Tamanho Comprimido (bytes)', 'Taxa de Compressão (Original/Comprimido)',
                  'Bits Per Pixel (bpp)', 'PSNR (dB)', 'SSIM', 'Tempo de Compressão (s)']
    
    # Escrever o cabeçalho do CSV inicialmente
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    collected_results_numeric = [] # Para coletar resultados com tipos numéricos para ordenação final

    print(f"\nIniciando {len(tasks_to_process)} tarefas de compressão com {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_params_str = {executor.submit(process_compression_task, task_args): 
                                f"{task_args['algo_type']}_{task_args['params']}" 
                                for task_args in tasks_to_process}
        
        for i, future in enumerate(as_completed(future_to_params_str)):
            params_str = future_to_params_str[future]
            try:
                result_numeric = future.result() # Dicionário com valores numéricos
                collected_results_numeric.append(result_numeric)

                # Formatar para escrita imediata no CSV
                row_to_write_immediately = result_numeric.copy()
                for key, val in row_to_write_immediately.items():
                    if isinstance(val, float):
                        if val == float('inf'): row_to_write_immediately[key] = 'inf'
                        elif key == 'SSIM': row_to_write_immediately[key] = f"{val:.4f}"
                        elif key in ['Bits Per Pixel (bpp)', 'Tempo de Compressão (s)']: row_to_write_immediately[key] = f"{val:.4f}"
                        else: row_to_write_immediately[key] = f"{val:.2f}" # PSNR, Taxa
                
                # Anexar ao CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row_to_write_immediately)
                print(f"({i+1}/{len(tasks_to_process)}) Resultado para: {params_str} salvo no CSV.")
            except Exception as exc:
                print(f"Tarefa {params_str} gerou uma exceção: {exc}")

    print(f"\nTodas as {len(collected_results_numeric)} tarefas concluídas. Reordenando o CSV final...")
    
    def sort_key(res_dict):
        ratio = res_dict.get('Taxa de Compressão (Original/Comprimido)', 0.0)
        comp_size = res_dict.get('Tamanho Comprimido (bytes)', float('inf'))
        ratio_val = float('inf') if ratio == float('inf') else float(ratio)
        return (-ratio_val, comp_size)

    sorted_results_for_final_csv = sorted(collected_results_numeric, key=sort_key)

    print(f"Reescrevendo CSV ordenado em: {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile: # Reabre em modo de escrita para sobrescrever
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res_numeric in sorted_results_for_final_csv:
            row_to_write = res_numeric.copy()
            for key, val in row_to_write.items(): # Reformatar números para string
                if isinstance(val, float):
                    if val == float('inf'): row_to_write[key] = 'inf'
                    elif key == 'SSIM': row_to_write[key] = f"{val:.4f}"
                    elif key in ['Bits Per Pixel (bpp)', 'Tempo de Compressão (s)']: row_to_write[key] = f"{val:.4f}"
                    else: row_to_write[key] = f"{val:.2f}"
            writer.writerow(row_to_write)
            
    print(f"\nTestes concluídos. Resultados finais salvos e ordenados em {csv_path}")
    return csv_path

# Funções de plotagem e CSV para modo de teste único (não paralelo)
def plot_compression_chart_single(labels, original_sizes_bytes, compressed_sizes_bytes, output_dir=None, filename_base=None):
    # ... (código de plot_compression_chart_single inalterado) ...
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    index = np.arange(len(labels))
    plt.bar(index, original_sizes_bytes, bar_width, label='Original (Pixels)')
    plt.bar(index + bar_width, compressed_sizes_bytes, bar_width, label='Comprimido')
    plt.xticks(index + bar_width / 2, labels)
    plt.ylabel('Tamanho (bytes)')
    plt.title('Comparação de Tamanhos de Compressão (Teste Único)')
    plt.legend(); plt.tight_layout()
    if output_dir and filename_base:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename_base}_compression_chart.png"))
    # plt.show(block=False); plt.pause(1); # Removido para evitar problemas em alguns ambientes
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compressão de imagem com Deflate e DWT, com análise detalhada e processamento paralelo.')
    parser.add_argument('input', help='Caminho da imagem de entrada (.nef, .png, .jpg, etc.)')
    parser.add_argument('--level', type=int, default=6, help='Nível de compressão zlib para Deflate (0-9)')
    parser.add_argument('--wavelet', default='haar', help='Wavelet para DWT (ex: haar, db1, coif1, sym2)')
    parser.add_argument('--dwt_level', type=int, default=1, help='Nível de decomposição DWT')
    parser.add_argument('--quant', type=float, default=10.0, help='Passo de quantização para DWT')
    parser.add_argument('--output_dir', default='output_compression', help='Diretório base para salvar resultados')
    parser.add_argument('--test_all', action='store_true', help='Testar automaticamente diferentes parâmetros em paralelo e salvar em subdiretório com timestamp')
    parser.add_argument('--save_streams', action='store_true', help='Salvar os fluxos de bytes comprimidos reais.')
    parser.add_argument('--workers', type=int, default=4, help='Número de workers para processamento paralelo em --test_all.')
    args = parser.parse_args()
    
    if args.test_all:
        if args.workers < 1: args.workers = 1 
        csv_results_path = run_parameter_tests(args.input, args.output_dir, args.save_streams, args.workers)
        print(f"Tabela de resultados detalhados e ordenados salva em: {csv_results_path}")
    else: # Modo de teste único (não paralelo)
        main_output_dir = args.output_dir
        os.makedirs(main_output_dir, exist_ok=True)
        images_output_dir_single = os.path.join(main_output_dir, "images_single_test"); os.makedirs(images_output_dir_single, exist_ok=True)
        plots_output_dir_single = os.path.join(main_output_dir, "plots_single_test"); os.makedirs(plots_output_dir_single, exist_ok=True)
        streams_output_dir_single = os.path.join(main_output_dir, "streams_single_test")
        if args.save_streams: os.makedirs(streams_output_dir_single, exist_ok=True)

        print(f"Carregando imagem: {args.input}")
        base_image_arr = load_image(args.input)
        original_pixel_bytes = base_image_arr.nbytes
        img_h, img_w = base_image_arr.shape; num_pixels = img_h * img_w
        print(f"Imagem carregada: {img_w}x{img_h} pixels, {original_pixel_bytes} bytes (float32)")

        results_for_csv = []

        print(f"\nExecutando compressão Deflate (nível {args.level})...")
        start_time_def = time.time()
        def_c_size, def_comp_stream = compress_deflate(base_image_arr, level=args.level)
        time_def = time.time() - start_time_def
        if args.save_streams:
            with open(os.path.join(streams_output_dir_single, f"image_deflate_level{args.level}.zlib"), 'wb') as f: f.write(def_comp_stream)
        results_for_csv.append({'Algoritmo': 'Deflate', 'Parâmetros': f'level={args.level}',
                                'Tamanho Original Pixels (bytes)': original_pixel_bytes, 'Dimensões (HxW)': f'{img_h}x{img_w}',
                                'Tamanho Comprimido (bytes)': def_c_size, 
                                'Taxa de Compressão (Original/Comprimido)': original_pixel_bytes / def_c_size if def_c_size > 0 else float('inf'),
                                'Bits Per Pixel (bpp)': (def_c_size * 8) / num_pixels if num_pixels > 0 else float('inf'),
                                'PSNR (dB)': float('inf'), 'SSIM': 1.0, 'Tempo de Compressão (s)': time_def})
        print(f"Deflate - Tamanho Comprimido: {def_c_size} bytes, Tempo: {time_def:.4f}s")

        print(f"\nExecutando compressão DWT (wavelet: {args.wavelet}, nível: {args.dwt_level}, quant: {args.quant})...")
        start_time_dwt = time.time()
        dwt_reconstructed_arr, dwt_c_size, dwt_comp_stream = compress_dwt(base_image_arr, wavelet=args.wavelet, level=args.dwt_level, quantization=args.quant)
        time_dwt = time.time() - start_time_dwt
        if args.save_streams:
            with open(os.path.join(streams_output_dir_single, f"image_dwt_{args.wavelet}_level{args.dwt_level}_quant{args.quant}.dwtz"), 'wb') as f: f.write(dwt_comp_stream)
        psnr_dwt_val = calculate_psnr_metric(base_image_arr, dwt_reconstructed_arr)
        ssim_dwt_val = structural_similarity(base_image_arr.astype(np.float32), dwt_reconstructed_arr.astype(np.float32), data_range=255.0)
        results_for_csv.append({'Algoritmo': 'DWT', 'Parâmetros': f'wavelet={args.wavelet},level={args.dwt_level},quant={args.quant}',
                                'Tamanho Original Pixels (bytes)': original_pixel_bytes, 'Dimensões (HxW)': f'{img_h}x{img_w}',
                                'Tamanho Comprimido (bytes)': dwt_c_size,
                                'Taxa de Compressão (Original/Comprimido)': original_pixel_bytes / dwt_c_size if dwt_c_size > 0 else float('inf'),
                                'Bits Per Pixel (bpp)': (dwt_c_size * 8) / num_pixels if num_pixels > 0 else float('inf'),
                                'PSNR (dB)': psnr_dwt_val, 'SSIM': ssim_dwt_val, 'Tempo de Compressão (s)': time_dwt})
        print(f"DWT - Tamanho Comprimido: {dwt_c_size} bytes, Tempo: {time_dwt:.4f}s, PSNR: {psnr_dwt_val:.2f} dB, SSIM: {ssim_dwt_val:.4f}")
        
        print("\nSalvando imagens e plots para DWT (teste único)...")
        fn_base_dwt_single = f"img_dwt_{args.wavelet}_level{args.dwt_level}_quant{args.quant}"
        save_images_for_task(base_image_arr, dwt_reconstructed_arr, images_output_dir_single, fn_base_dwt_single)
        plot_images_for_task(base_image_arr, dwt_reconstructed_arr, plots_output_dir_single, fn_base_dwt_single)
        plot_compression_chart_single(['Deflate', 'DWT'], [original_pixel_bytes, original_pixel_bytes], [def_c_size, dwt_c_size], 
                                   plots_output_dir_single, f"size_comp_deflate{args.level}_vs_dwt_q{args.quant}")
        
        csv_path_single = os.path.join(main_output_dir, "single_test_results_sorted.csv")
        print(f"\nSalvando resultados do teste único (ordenados) em CSV: {csv_path_single}")
        def sort_key_single(res_dict):
            ratio = res_dict.get('Taxa de Compressão (Original/Comprimido)', 0.0)
            comp_size = res_dict.get('Tamanho Comprimido (bytes)', float('inf'))
            ratio_val = float('inf') if ratio == float('inf') else float(ratio)
            return (-ratio_val, comp_size)
        results_for_csv_sorted = sorted(results_for_csv, key=sort_key_single)
        fieldnames_single = ['Algoritmo', 'Parâmetros', 'Tamanho Original Pixels (bytes)', 'Dimensões (HxW)',
                             'Tamanho Comprimido (bytes)', 'Taxa de Compressão (Original/Comprimido)',
                             'Bits Per Pixel (bpp)', 'PSNR (dB)', 'SSIM', 'Tempo de Compressão (s)']
        with open(csv_path_single, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_single)
            writer.writeheader()
            for res in results_for_csv_sorted:
                row_to_write = res.copy()
                for key, val in row_to_write.items():
                    if isinstance(val, float):
                        if val == float('inf'): row_to_write[key] = 'inf'
                        elif key == 'SSIM': row_to_write[key] = f"{val:.4f}"
                        elif key in ['Bits Per Pixel (bpp)', 'Tempo de Compressão (s)']: row_to_write[key] = f"{val:.4f}"
                        else: row_to_write[key] = f"{val:.2f}"
                writer.writerow(row_to_write)
        print(f"Resultados do teste único salvos e ordenados. Diretório de saída: {main_output_dir}")