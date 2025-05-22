#!/usr/bin/env python3
"""
Script principal para compressão de imagens (incluindo RAW .nef) 
utilizando Deflate e Discrete Wavelet Transform (DWT).

Permite executar testes únicos ou um conjunto abrangente de testes de parâmetros
em paralelo, salvando resultados e gerando gráficos de sumário.
"""
import argparse
import os
import time
from datetime import datetime
import numpy as np # Para np.nan em caso de falha no teste único

# Tenta definir o backend do Matplotlib antes de qualquer importação de pyplot.
# Isso é útil para ambientes sem GUI ou para evitar problemas em processamento paralelo.
try:
    import matplotlib
    matplotlib.use('Agg')
    print("Matplotlib backend definido para 'Agg'.")
except ImportError:
    print("Atenção: Matplotlib não encontrado ou backend 'Agg' não pôde ser definido.")
except Exception as e:
    print(f"Atenção: Erro ao definir backend 'Agg' para Matplotlib: {e}")

# Imports do projeto (após matplotlib.use)
import config # Constantes e configurações
from core import image_loader, compression, decompression, metrics
from plotting import individual_plots
from processing import task_manager
from utils import file_utils, csv_utils


def run_single_test_mode(args: argparse.Namespace, base_output_dir: str) -> None:
    """
    Executa o modo de teste único com os parâmetros fornecidos via CLI.
    """
    print(f"Modo de Teste Único - Diretório de Saída Principal: {base_output_dir}")
    
    # Criação de subdiretórios para o teste único
    images_out_dir = os.path.join(base_output_dir, config.DIR_IMAGES_SINGLE_TEST)
    plots_out_dir = os.path.join(base_output_dir, config.DIR_PLOTS_SINGLE_TEST)
    streams_out_dir = os.path.join(base_output_dir, config.DIR_STREAMS_SINGLE_TEST)

    file_utils.ensure_dir_exists(images_out_dir)
    file_utils.ensure_dir_exists(plots_out_dir)
    if args.save_streams:
        file_utils.ensure_dir_exists(streams_out_dir)

    print(f"Carregando imagem: {args.input}")
    try:
        base_image_arr = image_loader.load_image(args.input)
    except Exception as e:
        print(f"Erro crítico ao carregar imagem de entrada: {e}")
        return

    original_pixel_bytes = base_image_arr.nbytes
    img_h, img_w = base_image_arr.shape
    num_pixels = img_h * img_w
    print(f"Imagem carregada: {img_w}x{img_h} pixels, {original_pixel_bytes} bytes (dtype: {base_image_arr.dtype})")

    if args.save_streams:
        file_utils.save_stream(base_image_arr.tobytes(), streams_out_dir, config.ORIGINAL_IMAGE_STREAM_FILENAME)

    results_for_csv_single_test: list[dict] = []

    # --- Teste Deflate ---
    print(f"\nExecutando compressão Deflate (nível {args.level})...")
    start_time_def = time.time()
    try:
        def_c_size, def_comp_stream = compression.compress_deflate(base_image_arr, level=args.level)
        time_def = time.time() - start_time_def
        if args.save_streams:
            file_utils.save_stream(def_comp_stream, streams_out_dir, f"image_deflate_level{args.level}.zlib")
        
        res_def_single = {
            config.KEY_ALGORITHM: 'Deflate', 
            config.KEY_PARAMETERS: f'level={args.level}',
            config.KEY_WAVELET_NAME: '', # Vazio para Deflate
            config.KEY_RAW_PARAMS: {'level': args.level},
            config.KEY_ORIGINAL_SIZE_BYTES: original_pixel_bytes, 
            config.KEY_DIMENSIONS: f'{img_h}x{img_w}',
            config.KEY_COMPRESSED_SIZE_BYTES: def_c_size, 
            config.KEY_COMPRESSION_RATIO: original_pixel_bytes / def_c_size if def_c_size > 0 else float('inf'),
            config.KEY_BPP: (def_c_size * 8) / num_pixels if num_pixels > 0 and def_c_size > 0 else float('inf'),
            config.KEY_PSNR: float('inf'), 
            config.KEY_SSIM: 1.0, 
            config.KEY_COMPRESSION_TIME_S: time_def
        }
        results_for_csv_single_test.append(res_def_single)
        print(f"Deflate - Tamanho Comprimido: {def_c_size} bytes, Tempo: {time_def:.4f}s")
    except Exception as e:
        print(f"Erro durante a compressão Deflate: {e}")
        def_c_size = -1 # Indicar falha

    # --- Teste DWT ---
    print(f"\nExecutando compressão DWT (wavelet: {args.wavelet}, nível: {args.dwt_level}, quant: {args.quant})...")
    start_time_dwt = time.time()
    dwt_c_size = -1 # Inicializa para indicar falha
    try:
        # Nota: compress_dwt retorna (compressed_stream, metadata, compressed_size)
        # Precisamos chamar decompress_dwt para obter a imagem reconstruída.
        dwt_comp_stream, dwt_metadata, dwt_c_size_val = compression.compress_dwt(
            base_image_arr, wavelet=args.wavelet, level=args.dwt_level, quantization_step=args.quant
        )
        dwt_c_size = dwt_c_size_val # Atualiza o tamanho comprimido
        dwt_reconstructed_arr = decompression.decompress_dwt(dwt_comp_stream, dwt_metadata)
        time_dwt = time.time() - start_time_dwt

        if args.save_streams:
            stream_fn = f"image_dwt_{args.wavelet}_level{args.dwt_level}_quant{args.quant}.dwtz"
            file_utils.save_stream(dwt_comp_stream, streams_out_dir, stream_fn)
        
        psnr_dwt_val = metrics.calculate_psnr(base_image_arr, dwt_reconstructed_arr)
        ssim_dwt_val = metrics.calculate_ssim(base_image_arr, dwt_reconstructed_arr)
        
        res_dwt_single = {
            config.KEY_ALGORITHM: 'DWT', 
            config.KEY_PARAMETERS: f'wavelet={args.wavelet},level={args.dwt_level},quant={args.quant}',
            config.KEY_WAVELET_NAME: args.wavelet,
            config.KEY_RAW_PARAMS: {'wavelet': args.wavelet, 'level': args.dwt_level, 'quant': args.quant},
            config.KEY_ORIGINAL_SIZE_BYTES: original_pixel_bytes, 
            config.KEY_DIMENSIONS: f'{img_h}x{img_w}',
            config.KEY_COMPRESSED_SIZE_BYTES: dwt_c_size,
            config.KEY_COMPRESSION_RATIO: original_pixel_bytes / dwt_c_size if dwt_c_size > 0 else float('inf'),
            config.KEY_BPP: (dwt_c_size * 8) / num_pixels if num_pixels > 0 and dwt_c_size > 0 else float('inf'),
            config.KEY_PSNR: psnr_dwt_val, 
            config.KEY_SSIM: ssim_dwt_val, 
            config.KEY_COMPRESSION_TIME_S: time_dwt
        }
        results_for_csv_single_test.append(res_dwt_single)
        print(f"DWT - Tamanho Comprimido: {dwt_c_size} bytes, Tempo: {time_dwt:.4f}s, PSNR: {psnr_dwt_val:.2f} dB, SSIM: {ssim_dwt_val:.4f}")

        print("\nSalvando imagens e plots para DWT (teste único)...")
        fn_base_dwt_single = f"img_dwt_{args.wavelet}_level{args.dwt_level}_quant{args.quant}"
        individual_plots.save_reconstructed_image(dwt_reconstructed_arr, images_out_dir, fn_base_dwt_single)
        individual_plots.plot_image_comparison(
            base_image_arr, dwt_reconstructed_arr, plots_out_dir, fn_base_dwt_single,
            reconstructed_title=f'DWT ({args.wavelet}, L{args.dwt_level}, Q{args.quant})'
        )
    except Exception as e:
        print(f"Erro durante a compressão/descompressão DWT: {e}")
        # Se DWT falhar, dwt_c_size permanecerá -1 ou o valor antes da falha

    # --- Plot Comparativo (apenas se ambos os testes tiveram sucesso em obter tamanho) ---
    if def_c_size != -1 and dwt_c_size != -1:
        plot_labels = ['Deflate', 'DWT']
        plot_compressed_sizes = [def_c_size, dwt_c_size]
        chart_fn_base = f"size_comp_deflate{args.level}_vs_dwt_q{args.quant}"
        individual_plots.plot_single_test_compression_chart(
            plot_labels, original_pixel_bytes, plot_compressed_sizes,
            plots_out_dir, chart_fn_base
        )
    elif def_c_size == -1 and dwt_c_size == -1:
        print("\nAmbos os testes (Deflate e DWT) falharam. Nenhum gráfico de comparação de tamanho será gerado.")
    else:
        print("\nUm dos testes (Deflate ou DWT) falhou. Gráfico de comparação de tamanho não será completo.")


    # --- Salvar Resultados do Teste Único em CSV ---
    if results_for_csv_single_test:
        csv_path_single = os.path.join(base_output_dir, config.CSV_SINGLE_TEST_RESULTS_FILENAME)
        print(f"\nSalvando resultados do teste único (ordenados) em CSV: {csv_path_single}")
        csv_utils.sort_and_rewrite_csv(results_for_csv_single_test, csv_path_single, config.CSV_FIELDNAMES)
    else:
        print("\nNenhum resultado de teste único para salvar em CSV.")
        
    print(f"\nTeste(s) único(s) concluído(s). Diretório de saída: {base_output_dir}")


def main():
    """
    Função principal para executar o script de compressão de imagem.
    """
    parser = argparse.ArgumentParser(
        description='Compressão de imagem com Deflate e DWT, com análise detalhada e processamento paralelo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Mostra padrões na ajuda
    )
    parser.add_argument('input', help='Caminho da imagem de entrada (.nef, .png, .jpg, etc.)')
    
    # Argumentos para teste único (se --test_all não for usado)
    parser.add_argument('--level', type=int, default=6, choices=range(10), metavar='[0-9]',
                        help='Nível de compressão zlib para Deflate (0-9) no modo de teste único.')
    parser.add_argument('--wavelet', default='haar', 
                        help='Wavelet para DWT no modo de teste único (ex: haar, db4, bior2.2).')
    parser.add_argument('--dwt_level', type=int, default=1, 
                        help='Nível de decomposição DWT no modo de teste único.')
    parser.add_argument('--quant', type=float, default=10.0, 
                        help='Passo de quantização para DWT no modo de teste único.')
    
    # Argumentos gerais e para --test_all
    parser.add_argument('--output_dir', default='output_compression_results', 
                        help='Diretório base para salvar todos os resultados.')
    parser.add_argument('--test_all', action='store_true', 
                        help='Executar um conjunto abrangente de testes de parâmetros em paralelo. '
                             'Os parâmetros individuais (--level, --wavelet, etc.) são ignorados neste modo. '
                             'Os resultados são salvos em um subdiretório com timestamp.')
    parser.add_argument('--save_streams', action='store_true', 
                        help='Salvar os fluxos de bytes comprimidos reais (pode gerar muitos arquivos).')
    parser.add_argument('--workers', type=int, default=4, # Padrão para número de CPUs
                        help='Número de workers para processamento paralelo no modo --test_all.')
    
    args = parser.parse_args()

    if args.workers < 1:
        print("Número de workers deve ser pelo menos 1. Ajustando para 1.")
        args.workers = 1

    # Cria o diretório de saída base principal se não existir
    file_utils.ensure_dir_exists(args.output_dir)

    if args.test_all:
        timestamp = datetime.now().strftime(config.OUTPUT_DIR_TIMESTAMP_FORMAT)
        # Diretório específico para esta execução de --test_all
        test_all_output_dir = os.path.join(args.output_dir, f"{config.BASE_OUTPUT_DIR_NAME}_{timestamp}")
        file_utils.ensure_dir_exists(test_all_output_dir)
        
        print(f"Modo Testar Todos Ativado. Resultados em: {test_all_output_dir}")
        print(f"Carregando imagem base para testes: {args.input}")
        try:
            base_img_arr_for_tests = image_loader.load_image(args.input)
        except Exception as e:
            print(f"Erro crítico ao carregar imagem para --test_all: {e}")
            return

        csv_results_path = task_manager.run_parameter_tests(
            base_img_arr_for_tests,
            test_all_output_dir, 
            args.save_streams, 
            args.workers
        )
        print(f"Tabela de resultados detalhados e ordenados salva em: {csv_results_path}")
    else:
        # Modo de teste único usa o output_dir diretamente (ou um subdiretório dentro dele)
        # Para manter a consistência com --test_all, criamos um subdiretório para o teste único também.
        single_run_output_dir = os.path.join(args.output_dir, "single_test_run")
        file_utils.ensure_dir_exists(single_run_output_dir)
        run_single_test_mode(args, single_run_output_dir)

if __name__ == '__main__':
    main()
