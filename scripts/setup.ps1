# YOLO Object Detection - Windows Setup
# Conda 설치 및 NVIDIA GPU 환경을 전제로 합니다.

$ErrorActionPreference = "Stop"

$envName = "yolo-project"

# 프로젝트 루트로 이동
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Conda 환경 존재 여부 확인
$envList = conda env list --json | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    throw "Conda 환경 목록을 확인하지 못했습니다."
}

$envExists = @(
    $envList.envs | Where-Object {
        (Split-Path $_ -Leaf) -eq $envName
    }
).Count -gt 0

if (-not $envExists) {
    Write-Host "Creating Conda environment..."
    conda create -n $envName python=3.11 pip -y

    if ($LASTEXITCODE -ne 0) {
        throw "Conda 환경 생성에 실패했습니다."
    }
}

# Conda 환경에서 사용하는 Python 경로 확인
$envPaths = conda env list --json | ConvertFrom-Json
$envPath = $envPaths.envs |
    Where-Object { (Split-Path $_ -Leaf) -eq $envName } |
    Select-Object -First 1

if (-not $envPath) {
    throw "yolo-project 환경을 찾지 못했습니다."
}

$python = Join-Path $envPath "python.exe"

# GPU용 PyTorch 설치
Write-Host "Installing PyTorch CUDA 12.6..."
& $python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch 설치에 실패했습니다."
}

# 프로젝트 패키지 설치
Write-Host "Installing project requirements..."
& $python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "requirements.txt 설치에 실패했습니다."
}

# 설치 확인
Write-Host "Checking installation..."
& $python -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA available:', torch.cuda.is_available())"

if ($LASTEXITCODE -ne 0) {
    throw "설치 확인에 실패했습니다."
}

Write-Host "Setup complete!"