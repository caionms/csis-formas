# CSIS - Cloud Security Interoperable Society

## Visão Geral
O **CSIS** (Cloud Security Interoperable Society) é um sistema de vigilância universitária baseado em inteligência artificial para monitoramento em tempo real e detecção de ameaças. Ele integra múltiplos modelos de aprendizado de máquina para detectar eventos como comportamento suspeito, incêndios, alagamentos e reconhecimento de placas veiculares.

O sistema foi projetado para ser robusto, modular e escalável, garantindo interoperabilidade entre diferentes fontes de captura de vídeo e módulos de detecção.

## Requisitos
Para executar este projeto, é necessário ter **Python 3.12** ou superior instalado na máquina.

## Instalação

### 1. Instalar o Poetry
O gerenciamento de dependências do projeto é feito com **Poetry**. Para instalá-lo, execute:

```sh
pip install poetry
```

### 2. Instalar Dependências
Com o **Poetry** instalado, instale as dependências do projeto executando:

```sh
poetry install
```

### 3. Instalar PaddlePaddle e PyTorch
Além das dependências do Poetry, é necessário instalar manualmente o **PaddlePaddle** e o **PyTorch**, adequados ao seu ambiente.

#### 3.1 Instalar PyTorch
Para instalar a versão correspondente ao seu sistema operacional e hardware, siga as instruções no site oficial do PyTorch:
[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

#### 3.2 Instalar PaddlePaddle
Para instalar o **PaddlePaddle**, siga as instruções do site oficial conforme seu sistema:
[PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/macos-pip_en.html)

### 4. Configuração de Segredos
Para permitir o download dos modelos armazenados no Dropbox, é necessário configurar um arquivo de segredos.

1. Crie um arquivo `.secrets.local.toml` na raiz do projeto com base no modelo disponível em `.secrets.toml.example`.
2. Defina os parâmetros necessários para autenticação e acesso ao Dropbox.
3. Os valores exigidos são sigilosos e devem ser solicitados a um membro do **FORMAS**.

## Execução
O projeto pode ser executado por meio dos módulos disponíveis em:

```sh
application/use_cases/
```

### Rodando o Módulo de Detecção via Captura de Janela do Windows
Se deseja executar o módulo de detecção via captura de janela no Windows (`application/use_cases/run_window_capture.py`), é necessário instalar o **PyWinCtl**:

```sh
pip install PyWinCtl
```

Esse módulo permite a interação com janelas do sistema operacional para captura de frames em tempo real.

### Parâmetros de Execução
Os parâmetros de execução podem ser configurados diretamente na chamada da função `main`. O principal parâmetro a ser ajustado é `detection_type`, que define o tipo de detecção a ser realizado.

### Tipos de Detecção Disponíveis
O **CSIS** suporta os seguintes tipos de detecção, definidos na enumeração `DetectionTypeEnum`:

```python
from enum import Enum

class DetectionTypeEnum(Enum):
    """Enumeração de tipos de detecção."""
    
    PLATE_RECOGNITION = "PLATE_RECOGNITION"
    """Reconhecimento de placas veiculares."""
    
    SUSPICIOUS_PRESENCE = "SUSPICIOUS_PRESENCE"
    """Detecção de presença suspeita na câmera."""
    
    SUSPICIOUS_PROXIMITY_TO_VEHICLE = "SUSPICIOUS_PROXIMITY_TO_VEHICLE"
    """Detecção de comportamento suspeito por proximidade a veículos."""
    
    SUSPICIOUS_PROXIMITY_WITH_POSE = "SUSPICIOUS_PROXIMITY_WITH_POSE"
    """Detecção de comportamento suspeito considerando pose corporal."""
    
    PUBLIC_SAFETY = "PUBLIC_SAFETY"
    """Detecção de ameaças à segurança pública."""
    
    FIRE_SMOKE_DETECTION = "FIRE_SMOKE_DETECTION"
    """Detecção de fogo e fumaça."""
    
    FLOOD_DETECTION = "FLOOD_DETECTION"
    """Detecção de alagamentos."""
    
    WEAPON_DETECTION = "WEAPON_DETECTION"
    """Detecção de armas de fogo e armas brancas."""
    
    GRAFFITI_SPRAY_DETECTION = "GRAFFITI_SPRAY_DETECTION"
    """Detecção de pichações."""
```

Para executar um tipo específico de detecção, basta definir o valor adequado do `detection_type`.

## Contribuição
Caso queira contribuir com o projeto, siga as diretrizes padrão de desenvolvimento, garantindo código modular, bem documentado e aderente às boas práticas do **Zen of Python** e **PEP 8**.

## Licença
Este projeto é distribuído sob a licença **GNU General Public License (GPL) v3.0**.
