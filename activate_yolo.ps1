# Activate YOLOv8 environment
$env:CONDA_EXE = "D:\conda\Scripts\conda.exe"
Import-Module "D:\conda\shell\condabin\Conda.psm1"
conda activate yolov8

Write-Host "YOLOv8 environment activated!" -ForegroundColor Green
Write-Host "You can now run your Python scripts." -ForegroundColor Green
