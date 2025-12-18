Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Starting Training to 50 Epochs" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Current: Epoch 39, WER 0.50" -ForegroundColor Yellow
Write-Host "Target: 50 epochs (11 more epochs)" -ForegroundColor Yellow
Write-Host "Estimated time: ~55 minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 50 --batch_size 8 --lr 1e-3

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green

