@echo off
echo ========================================================================
echo Starting Training to 50 Epochs
echo ========================================================================
echo.
echo Current: Epoch 39, WER 0.50
echo Target: 50 epochs (11 more epochs)
echo Estimated time: ~55 minutes
echo.
echo ========================================================================
echo.

cd /d "%~dp0"
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 50 --batch_size 8 --lr 1e-3

echo.
echo ========================================================================
echo Training Complete!
echo ========================================================================
pause

