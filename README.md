# Análise Comparativa de Compressão de Dados: Deflate vs. DWT

Este repositório contém o código-fonte para o artigo **"Compressão de Dados: Uma Análise Comparativa entre Algoritmos Com Perda e Sem Perda"**, desenvolvido na disciplina de Modelagem Estatística e Programação do Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP).

O projeto implementa e analisa dois algoritmos de compressão de imagem:
1.  **Deflate (Sem Perda)**: Uma combinação de LZ77 e codificação Huffman, a mesma técnica usada em formatos como PNG e GZIP.
2.  **Discrete Wavelet Transform (DWT) (Com Perda)**: Uma técnica moderna que serve de base para o formato JPEG 2000, oferecendo alta compressão com boa qualidade visual.

O objetivo é fornecer uma ferramenta prática e teórica para comparar a eficiência, a qualidade e as características de desempenho desses dois métodos sob diversas condições.

**[Acesse o artigo completo aqui (PDF)](https://github.com/Ashu11-A/DataCompression-2025.1/tree/main/article/Artigo%20Compress%C3%A3o%20de%20Imagens%20%5BLaTeX%5D.pdf)**
## ✨ Principais Funcionalidades

-   **Compressão Dual**: Implementa tanto o algoritmo Deflate (sem perdas) quanto o DWT (com perdas) para uma comparação direta.
-   **Suporte a Imagens RAW**: Carrega e processa imagens de formatos padrão (PNG, JPG) e também arquivos RAW de câmeras (`.nef`), convertendo-os para uma análise consistente.
-   **Testes Parametrizáveis**: Permite executar testes únicos com parâmetros específicos ou um lote abrangente de testes em paralelo.
-   **Processamento Paralelo**: O modo de teste em lote (`--test_all`) utiliza múltiplos núcleos de CPU para acelerar a análise de centenas de combinações de parâmetros.
-   **Métricas de Avaliação**: Calcula automaticamente métricas essenciais de qualidade e eficiência:
    -   PSNR (Peak Signal-to-Noise Ratio)
    -   SSIM (Structural Similarity Index)
    -   Taxa de Compressão
    -   BPP (Bits Per Pixel)
-   **Geração de Resultados**: Salva todos os resultados em arquivos CSV detalhados e gera automaticamente:
    -   Gráficos de sumário que comparam o desempenho (ex: PSNR vs. BPP) para cada wavelet.
    -   Imagens reconstruídas para análise visual.
    -   Plots comparativos (Original vs. Reconstruída).

## 🚀 Instalação

Este projeto utiliza **`uv`** para gerenciamento de pacotes e ambientes virtuais.

1.  **Instale o `uv` (se ainda não o tiver):**
    ```bash
    # Use pip para instalar uv globalmente
    pip install uv
    ```

2.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/DataCompression-2025.1.git
    cd DataCompression-2025.1
    ```

3.  **Crie um ambiente virtual e instale as dependências com `uv`:**
    ```bash
    # Cria o ambiente virtual no diretório .venv
    uv venv

    # Ative o ambiente
    # Linux/macOS
    source .venv/bin/activate
    # Windows
    # .\.venv\Scripts\activate

    # Instala as dependências do pyproject.toml
    uv pip install .
    ```

## ⚙️ Como Usar

O script principal (`src/main.py`) pode ser executado em dois modos: teste único ou teste em lote.

### Modo de Teste Único

Execute uma comparação direta entre Deflate e DWT com parâmetros específicos.

```bash
python src/main.py "caminho/para/sua/imagem.nef" --level 9 --wavelet "db4" --dwt_level 4 --quant 25.0
```

-   `input`: Caminho para a imagem de entrada.
-   `--level`: Nível de compressão para Deflate (padrão: 6).
-   `--wavelet`: Tipo de wavelet para DWT (padrão: 'haar').
-   `--dwt_level`: Nível de decomposição para DWT (padrão: 1).
-   `--quant`: Passo de quantização para DWT (padrão: 10.0).

Os resultados serão salvos no diretório `output_compression_results/single_test_run/`.

### Modo de Teste em Lote (Completo)

Execute uma análise abrangente com todas as combinações de parâmetros definidas em `src/config.py`. Este modo é ideal para gerar os dados para os gráficos de sumário.

```bash
python src/main.py "caminho/para/sua/imagem.nef" --test_all --workers 8 --save_streams
```

-   `--test_all`: Ativa o modo de teste em lote.
-   `--workers`: Define o número de processos paralelos a serem usados (padrão: 4).
-   `--save_streams`: (Opcional) Salva os arquivos de bytes comprimidos.

Os resultados serão salvos em um subdiretório com data e hora dentro de `output_compression_results/`, por exemplo, `output_compression_results/comp_test_20250715_160300/`.

## 📂 Estrutura do Projeto

```
.
├── article/
│   └── Artigo Compressão de Imagens [LaTeX].pdf
├── src/
│   ├── core/
│   ├── plotting/
│   ├── processing/
│   ├── utils/
│   ├── config.py
│   └── main.py
├── output_compression_results/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

Copyright (c) 2025 Matheus N. Biolowons and Guilherme A. Montalvão

## ✍️ Autores e Agradecimentos

Este projeto foi desenvolvido como parte do trabalho acadêmico de **Matheus N. Biolowons** e **Guilherme A. Montalvão** no **Instituto Brasileiro de Ensino, Desenvolvimento e Pesquisa (IDP)**.

Agradecemos ao professor **Felipe Dias** pela orientação e suporte durante a disciplina de Modelagem Estatística e Programação!