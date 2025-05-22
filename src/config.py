"""
Módulo de configuração para o projeto de compressão de imagens.
Armazena constantes como nomes de colunas CSV, chaves de dicionário,
parâmetros padrão e configurações de plotagem.
"""
import numpy as np

# --- Constantes para Chaves de Dicionário de Resultados ---
KEY_ALGORITHM = 'Algoritmo'
KEY_PARAMETERS = 'Parâmetros'
KEY_WAVELET_NAME = 'wavelet_name' # Específico para DWT, usado para agrupamento
KEY_RAW_PARAMS = 'raw_params'
KEY_ORIGINAL_SIZE_BYTES = 'Tamanho Original Pixels (bytes)'
KEY_DIMENSIONS = 'Dimensões (HxW)'
KEY_COMPRESSED_SIZE_BYTES = 'Tamanho Comprimido (bytes)'
KEY_COMPRESSION_RATIO = 'Taxa de Compressão (Original/Comprimido)'
KEY_BPP = 'Bits Per Pixel (bpp)'
KEY_PSNR = 'PSNR (dB)'
KEY_SSIM = 'SSIM'
KEY_COMPRESSION_TIME_S = 'Tempo de Compressão (s)'

# --- Nomes de Colunas para o Arquivo CSV ---
CSV_FIELDNAMES = [
    KEY_ALGORITHM,
    KEY_PARAMETERS,
    KEY_WAVELET_NAME,
    KEY_RAW_PARAMS,
    KEY_ORIGINAL_SIZE_BYTES,
    KEY_DIMENSIONS,
    KEY_COMPRESSED_SIZE_BYTES,
    KEY_COMPRESSION_RATIO,
    KEY_BPP,
    KEY_PSNR,
    KEY_SSIM,
    KEY_COMPRESSION_TIME_S
]

# --- Parâmetros Padrão para Testes ---
DEFAULT_DEFLATE_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_DWT_WAVELETS = ['haar', 'db1', 'db4', 'sym2', 'coif1', 'bior2.2']
DEFAULT_DWT_LEVELS = list(range(2, 31, 2))
DEFAULT_DWT_QUANT_STEPS = list(range(5, 101, 5))

# --- Configurações de Processamento de Imagem ---
# Coeficientes para conversão RGB para Escala de Cinza (Luminância ITU-R BT.709)
RGB_TO_GRAY_COEFFS = [0.2126, 0.7152, 0.0722]
# Tipo de dados para coeficientes DWT quantizados
DWT_QUANTIZED_COEFF_DTYPE = np.int16
# Valor máximo de pixel para cálculo de PSNR (para imagens de 8 bits)
MAX_PIXEL_VALUE_8BIT = 255.0

# --- Configurações de Plotagem ---
# Paleta de marcadores para gráficos de sumário DWT
DWT_SUMMARY_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
# Paleta de cores para gráficos de sumário DWT
DWT_SUMMARY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
SUMMARY_PLOT_FIGSIZE = (12, 7)
DWT_SUMMARY_PLOT_FIGSIZE = (20, 10) # Largura aumentada
INDIVIDUAL_PLOT_FIGSIZE = (12, 6)
SINGLE_TEST_COMPRESSION_CHART_FIGSIZE = (8, 5)

# --- Nomes de Diretórios de Saída ---
DIR_IMAGES = "images"
DIR_SUMMARY_PLOTS = "summary_plots"
DIR_INDIVIDUAL_PLOTS = "individual_dwt_plots"
DIR_STREAMS = "streams"
DIR_IMAGES_SINGLE_TEST = "images_single_test"
DIR_PLOTS_SINGLE_TEST = "plots_single_test"
DIR_STREAMS_SINGLE_TEST = "streams_single_test"

# --- Outras Constantes ---
OUTPUT_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
BASE_OUTPUT_DIR_NAME = "comp_test"
CSV_RESULTS_FILENAME = "compression_results_sorted.csv"
CSV_SINGLE_TEST_RESULTS_FILENAME = "single_test_results_sorted.csv"
ORIGINAL_IMAGE_STREAM_FILENAME = "image_original.data"
