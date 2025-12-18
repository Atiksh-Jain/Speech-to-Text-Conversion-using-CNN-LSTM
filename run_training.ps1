# Training Script - Run this to start training
Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host "Starting Training to 40 Epochs"
Write-Host ("=" * 70)
Write-Host ""

# Change to project directory
Set-Location "C:\Users\Lenovo\Desktop\vtu-vtu\stt_cnn_lstm"

# Run training
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 40 --batch_size 8 --lr 1e-3

Write-Host ""
Write-Host ("=" * 70)
Write-Host "Training Complete!"
Write-Host ("=" * 70)

