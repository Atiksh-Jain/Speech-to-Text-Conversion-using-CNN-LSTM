@echo off
cd /d "%~dp0"
echo ================================================================================
echo Starting Training from stt_cnn_lstm directory
echo ================================================================================
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 350 --batch_size 8 --lr 1e-3
pause

