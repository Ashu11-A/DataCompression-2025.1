"""
Módulo para gerar gráficos de sumário de desempenho dos algoritmos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import config # Para constantes de figura e chaves
from utils.file_utils import ensure_dir_exists # Importando de utils

def plot_deflate_summary_chart(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Plota um gráfico de sumário para o desempenho do algoritmo Deflate
    em diferentes níveis de compressão.

    Args:
        results (List[Dict[str, Any]]): Lista de dicionários de resultados para Deflate.
        output_dir (str): Diretório para salvar o gráfico.
    """
    if not results:
        print("Aviso (Deflate Summary): Nenhum resultado fornecido para plotagem.")
        return
    
    ensure_dir_exists(output_dir)

    levels = []
    compressed_sizes = []
    ratios = []
    # times = [] # Descomente se quiser plotar o tempo

    # Extrair e ordenar dados por nível para plotagem consistente
    # Assume que 'Parâmetros' é como 'level=X'
    try:
        sorted_results = sorted(results, key=lambda r: int(r[config.KEY_PARAMETERS].split('=')[1]))
    except (ValueError, KeyError, IndexError):
        print("Aviso (Deflate Summary): Erro ao ordenar resultados. Verifique o formato dos parâmetros.")
        sorted_results = results # Tenta plotar com os dados não ordenados

    for res in sorted_results:
        try:
            levels.append(int(res[config.KEY_PARAMETERS].split('=')[1]))
            compressed_sizes.append(res[config.KEY_COMPRESSED_SIZE_BYTES])
            ratios.append(res[config.KEY_COMPRESSION_RATIO])
            # times.append(res[config.KEY_COMPRESSION_TIME_S])
        except (ValueError, IndexError, KeyError) as e:
            print(f"Aviso (Deflate Summary): Não foi possível parsear dados para o sumário: {res}. Erro: {e}")
            continue

    if not levels:
        print("Aviso (Deflate Summary): Nenhum dado válido para plotar após o parsing.")
        return

    fig, ax1 = plt.subplots(figsize=config.SUMMARY_PLOT_FIGSIZE)

    color = 'tab:red'
    ax1.set_xlabel('Nível de Compressão Deflate')
    ax1.set_ylabel('Tamanho Comprimido (bytes)', color=color)
    ax1.plot(levels, compressed_sizes, color=color, marker='o', linestyle='-', label='Tamanho Comprimido')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(levels) # Garante que todos os níveis testados apareçam

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Taxa de Compressão', color=color)
    ax2.plot(levels, ratios, color=color, marker='x', linestyle='--', label='Taxa de Compressão')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle('Sumário de Desempenho Deflate')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    lines, labels_ax1 = ax1.get_legend_handles_labels()
    lines2, labels_ax2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels_ax1 + labels_ax2, loc='center right')
    
    plt.grid(True, linestyle=':')
    plot_path = os.path.join(output_dir, "summary_deflate_performance.png")
    try:
        plt.savefig(plot_path)
        print(f"Gráfico de sumário Deflate salvo em: {plot_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico de sumário Deflate: {e}")
    finally:
        plt.close(fig)

def plot_dwt_wavelet_summary_chart(wavelet_name: str, results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Plota um gráfico de sumário (PSNR vs BPP) para uma wavelet DWT específica,
    com diferentes níveis de DWT representados por marcadores/cores.

    Args:
        wavelet_name (str): Nome da wavelet (ex: 'haar').
        results (List[Dict[str, Any]]): Lista de dicionários de resultados para esta wavelet.
        output_dir (str): Diretório para salvar o gráfico.
    """
    if not results:
        print(f"Aviso (DWT Summary - {wavelet_name}): Nenhum resultado fornecido.")
        return

    ensure_dir_exists(output_dir)

    bpps = []
    psnrs = []
    dwt_levels_from_params = [] # Para forma e cor dos pontos

    for res in results:
        try:
            # Filtrar PSNR infinito, pois não pode ser plotado numericamente
            psnr_val = res[config.KEY_PSNR]
            if psnr_val == float('inf'):
                continue # Pula este ponto

            bpps.append(res[config.KEY_BPP])
            psnrs.append(psnr_val)
            
            # Parsear parâmetros como "wavelet=haar,level=4,quant=10"
            params_parts = res[config.KEY_PARAMETERS].split(',')
            level_val = int(next(p for p in params_parts if 'level=' in p).split('=')[1])
            dwt_levels_from_params.append(level_val)
        except (ValueError, IndexError, KeyError, StopIteration) as e:
            print(f"Aviso (DWT Summary - {wavelet_name}): Não foi possível parsear dados: {res}. Erro: {e}")
            continue
    
    if not bpps: # Se todos os PSNRs eram 'inf' ou houve erros de parsing
        print(f"Aviso (DWT Summary - {wavelet_name}): Nenhum dado válido para plotar (BPPs vazios ou PSNRs infinitos).")
        return

    unique_dwt_levels = sorted(list(set(dwt_levels_from_params)))
    
    level_to_marker = {
        level: config.DWT_SUMMARY_MARKERS[i % len(config.DWT_SUMMARY_MARKERS)] 
        for i, level in enumerate(unique_dwt_levels)
    }
    level_to_color = {
        level: config.DWT_SUMMARY_COLORS[i % len(config.DWT_SUMMARY_COLORS)]
        for i, level in enumerate(unique_dwt_levels)
    }

    plt.figure(figsize=config.DWT_SUMMARY_PLOT_FIGSIZE)
    scatter_handles = []

    for level_val in unique_dwt_levels:
        level_bpps = [bpps[i] for i, l_param in enumerate(dwt_levels_from_params) if l_param == level_val]
        level_psnrs = [psnrs[i] for i, l_param in enumerate(dwt_levels_from_params) if l_param == level_val]
        
        if level_bpps: # Apenas plota se houver dados para este nível
            handle = plt.scatter(level_bpps, level_psnrs, 
                                 color=level_to_color[level_val],
                                 marker=level_to_marker[level_val],
                                 label=f'Level {level_val}', 
                                 alpha=0.85, s=100) # s é o tamanho do marcador
            scatter_handles.append(handle)

    if not scatter_handles:
        print(f"Aviso (DWT Summary - {wavelet_name}): Nenhum dado foi efetivamente plotado.")
        plt.close()
        return

    plt.xlabel('Bits Por Pixel (bpp)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'Sumário DWT: Wavelet "{wavelet_name}" (PSNR vs BPP)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Legenda para marcadores (e cores)
    plt.legend(title='Nível DWT', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98]) # Ajuste para legenda externa

    plot_path = os.path.join(output_dir, f"summary_dwt_{wavelet_name}_performance.png")
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Gráfico de sumário DWT para {wavelet_name} salvo em: {plot_path}")
    except Exception as e:
        print(f"Erro ao salvar o gráfico de sumário DWT {plot_path}: {e}")
    finally:
        plt.close()
