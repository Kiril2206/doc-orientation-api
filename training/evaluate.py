import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import create_splits
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Document Orientation Classifier")
    parser.add_argument('--model-path', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--data-source', type=str, default='hf', choices=['hf', 'local'])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--num-images', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='./output')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Preparing test dataset...")
    _, _, test_ds = create_splits(
        source=args.data_source, 
        data_dir=args.data_dir, 
        num_images=args.num_images
    )
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    class_names = ['0°', '90°', '180°', '270°']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    main()
