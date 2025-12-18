@echo off
cd /d "%~dp0"
echo ========================================================================
echo Starting Training to 350 Epochs
echo ========================================================================
echo.
echo Current: Epoch 50
echo Target: 350 epochs (300 more)
echo Early stopping: 7 epochs without improvement
echo.
echo ========================================================================
echo.

python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 350 --batch_size 8 --lr 1e-3

echo.
echo ========================================================================
echo Training Complete or Stopped
echo ========================================================================
pause

