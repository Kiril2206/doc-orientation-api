# Train ResNet-18 (v1 model) on the universal global dataset for the Hybrid OCR pipeline
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Training ResNet-18 for Hybrid (ML + OCR) Pipeline" -ForegroundColor Cyan
Write-Host " Dataset: data/v2_universal_4k (Global Multilingual Mix)" -ForegroundColor Cyan
Write-Host " Backbone: ResNet-18 | Input Size: 224px | Device: CUDA" -ForegroundColor Cyan
Write-Host " Target ONNX: model/orientation_model.onnx" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

python.exe -u -m training.train `
  --manifest data/v2_universal_4k/manifest.jsonl `
  --backbone resnet18 `
  --input-size 224 `
  --model-version v1 `
  --epochs 15 `
  --batch-size 16 `
  --device cuda `
  --export-onnx `
  --onnx-output-path model/orientation_model.onnx `
  --output-dir output/v1_resnet18_universal

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " SUCCESS! ResNet-18 trained and exported to model/orientation_model.onnx" -ForegroundColor Green
Write-Host " The Hybrid pipeline (v1 ResNet-18 + OCR) is now updated with the new dataset!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
