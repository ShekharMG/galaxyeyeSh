@echo off
echo ========================================================
echo SAR-EO Change Detection - Full Pipeline
echo ========================================================

echo.
echo [1/3] Activating Virtual Environment...
call venv\Scripts\activate

echo.
echo [2/3] Starting Training...
python train.py --config config.yaml

echo.
echo [3/3] Running Evaluation on Test Set...
python eval.py --data_path ./ --weights checkpoints/best_model.pth --split test

echo.
echo ========================================================
echo Pipeline Complete! Results are saved in eval_results/
echo ========================================================
pause
