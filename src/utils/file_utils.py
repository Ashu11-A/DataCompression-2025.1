"""
Módulo com funções utilitárias para manipulação de arquivos e diretórios.
"""
import os

def ensure_dir_exists(dir_path: str) -> None:
    """
    Garante que um diretório exista. Se não existir, cria-o.

    Args:
        dir_path (str): Caminho do diretório a ser verificado/criado.
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True) # exist_ok=True evita erro se o dir for criado entre o check e o makedirs
            print(f"Diretório criado: {dir_path}")
        except OSError as e:
            print(f"Erro ao criar diretório {dir_path}: {e}")
            # Considerar levantar a exceção dependendo da criticidade
            # raise

def save_stream(stream_data: bytes, output_dir: str, filename: str) -> None:
    """
    Salva um stream de bytes em um arquivo.

    Args:
        stream_data (bytes): Os dados binários a serem salvos.
        output_dir (str): Diretório onde o arquivo será salvo.
        filename (str): Nome do arquivo.
    """
    ensure_dir_exists(output_dir)
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, 'wb') as f:
            f.write(stream_data)
        # print(f"Stream salvo em: {file_path}") # Opcional: pode ser muito verboso
    except IOError as e:
        print(f"Erro ao salvar stream em {file_path}: {e}")

