# Automated pipeline to prepare the balanced universal dataset and train Model v2
$ErrorActionPreference = "Stop"

if (!(Test-Path "data/v2_universal_4k/manifest.jsonl")) {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " STEP 1: Preparing Realistic Balanced Universal Dataset (~4.1k pages)" -ForegroundColor Cyan
    Write-Host " Proportions: 60% Corporate/Academic, 30% World Scripts (Cyrillic, Arabic, Indic, CJK), 6% Receipts, 4% Forms" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    python.exe -u -m training.prepare_universal `
      --output-dir data/v2_universal_4k `
      --doclaynet 2500 `
      --cord 250
} else {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " STEP 1: Universal Dataset already prepared (4,093 pages). Skipping to training!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Cyan
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " STEP 2: Training Universal Model v2 on GPU (GTX 1050 Ti)" -ForegroundColor Green
Write-Host " Backbone: EfficientNet-B0 (384px) | Epochs: 15 | Batch size: 8" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

python.exe -u -m training.train `
  --manifest data/v2_universal_4k/manifest.jsonl `
  --backbone efficientnet_b0 `
  --input-size 384 `
  --epochs 15 `
  --batch-size 8 `
  --device cuda `
  --export-onnx `
  --onnx-output-path model/orientation_model_v2.onnx `
  --output-dir output/v2_universal_4k

Write-Host "`n==========================================================" -ForegroundColor Yellow
Write-Host " SUCCESS! Universal model trained and exported to model/orientation_model_v2.onnx" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Yellow
