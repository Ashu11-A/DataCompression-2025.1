"""
Módulo para gerar plots de imagens individuais e comparações simples.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import config
from utils.file_utils import ensure_dir_exists 

def save_reconstructed_image(reconstructed_arr: np.ndarray, output_dir: str, filename_base: str) -> None:
    """
    Salva a imagem reconstruída como um arquivo PNG.

    Args:
        reconstructed_arr (np.ndarray): Array da imagem reconstruída.
        output_dir (str): Diretório para salvar a imagem.
        filename_base (str): Base do nome do arquivo (sem extensão).
    """
    ensure_dir_exists(output_dir)
    reconstructed_uint8 = reconstructed_arr.clip(0, 255).astype(np.uint8)
    try:
        Image.fromarray(reconstructed_uint8).save(os.path.join(output_dir, f"{filename_base}_reconstructed.png"))
    except Exception as e:
        print(f"Erro ao salvar imagem reconstruída {filename_base}: {e}")

def plot_image_comparison(original_arr: np.ndarray, reconstructed_arr: np.ndarray, 
                          output_dir: str, filename_base: str, 
                          reconstructed_title: str = 'Imagem Reconstruída') -> None:
    """
    Plota a imagem original e a reconstruída lado a lado e salva a figura.

    Args:
        original_arr (np.ndarray): Array da imagem original.
        reconstructed_arr (np.ndarray): Array da imagem reconstruída.
        output_dir (str): Diretório para salvar o plot.
        filename_base (str): Base do nome do arquivo do plot.
        reconstructed_title (str): Título para a imagem reconstruída no plot.
    """
    ensure_dir_exists(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=config.INDIVIDUAL_PLOT_FIGSIZE)
    
    axes[0].imshow(original_arr, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed_arr, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(reconstructed_title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{filename_base}_comparison.png")
    try:
        plt.savefig(plot_path)
    except Exception as e:
        print(f"Erro ao salvar plot de comparação {filename_base}: {e}")
    finally:
        plt.close(fig)

def plot_single_test_compression_chart(
    labels: List[str], 
    original_size_bytes: int, # Assumindo que o tamanho original é o mesmo para os testes comparados
    compressed_sizes_bytes: List[int], 
    output_dir: str, 
    filename_base: str
) -> None:
    """
    Plota um gráfico de barras comparando o tamanho original com os tamanhos comprimidos
    para um conjunto limitado de testes (ex: Deflate vs DWT com um conjunto de params).

    Args:
        labels (List[str]): Rótulos para cada barra de compressão (ex: ['Deflate', 'DWT']).
        original_size_bytes (int): Tamanho da imagem original em bytes.
        compressed_sizes_bytes (List[int]): Lista de tamanhos comprimidos em bytes.
        output_dir (str): Diretório para salvar o gráfico.
        filename_base (str): Base do nome do arquivo do gráfico.
    """
    ensure_dir_exists(output_dir)
    
    num_tests = len(labels)
    if num_tests != len(compressed_sizes_bytes):
        print("Erro: Número de rótulos e tamanhos comprimidos não coincide para o gráfico.")
        return

    plt.figure(figsize=config.SINGLE_TEST_COMPRESSION_CHART_FIGSIZE)
    bar_width = 0.35
    index = np.arange(num_tests)

    # Barras para o tamanho original (repetidas para cada teste para comparação)
    plt.bar(index, [original_size_bytes] * num_tests, bar_width, label='Original (Pixels)', color='skyblue')
    # Barras para os tamanhos comprimidos
    plt.bar(index + bar_width, compressed_sizes_bytes, bar_width, label='Comprimido', color='salmon')

    plt.xlabel('Método de Compressão')
    plt.ylabel('Tamanho (bytes)')
    plt.title('Comparação de Tamanhos de Compressão (Teste Único)')
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{filename_base}_compression_chart.png")
    try:
        plt.savefig(plot_path)
        print(f"Gráfico de comparação de teste único salvo em: {plot_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico de comparação de teste único {filename_base}: {e}")
    finally:
        plt.close()
