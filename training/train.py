import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
try:
    from dataset import create_splits
except ImportError:
    from training.dataset import create_splits

try:
    from export_onnx import export_model
except ImportError:
    from training.export_onnx import export_model

def get_args():
    parser = argparse.ArgumentParser(description="Train Document Orientation Classifier")
    parser.add_argument('--data-source', default='hf_mixed',
                        choices=['hf_mixed', 'hf_doclaynet', 'hf_cord', 'hf_rvlcdip', 'hf', 'local'],
                        help="DocLayNet + CORD by default; see docs/DATASETS.md")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--num-images', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze-epochs', type=int, default=3, help="Number of epochs to freeze early layers")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--pretrained', action='store_true', default=True, help="Use ImageNet pre-trained weights for transfer learning")
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help="Train from scratch without pre-trained weights")
    parser.add_argument('--export-onnx', action='store_true', help="Automatically export best model to ONNX after training")
    parser.add_argument('--onnx-output-path', type=str, default=os.path.join('model', 'orientation_model.onnx'), help="Destination path for ONNX export")
    return parser.parse_args()

def set_parameter_requires_grad(model, layers, requires_grad=False):
    for name, param in model.named_parameters():
        for layer in layers:
            if name.startswith(layer):
                param.requires_grad = requires_grad

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Preparing datasets...")
    train_ds, val_ds = create_splits(
        source=args.data_source, 
        data_dir=args.data_dir, 
        num_images=args.num_images,
        seed=args.seed,
        splits=("train", "validation"),
    )
    
    # We use num_workers=0 to avoid multiprocessing issues on some Windows systems unless specified
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    manifest = dict(vars(args))
    manifest["split_sizes"] = {"train": len(train_ds), "validation": len(val_ds)}
    manifest["label_rotation"] = "counterclockwise"
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Model
    if args.pretrained:
        print("Using ResNet-18 with ImageNet pre-trained weights for transfer learning...")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        print("Using ResNet-18 initialized from scratch (random weights)...")
        model = resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(device)
    
    # Early layers to freeze initially (only for transfer learning)
    early_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    if args.pretrained and args.freeze_epochs > 0:
        set_parameter_requires_grad(model, early_layers, requires_grad=False)
    
    # Optimizer & Scheduler & Loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Unfreeze after freeze_epochs (only when using pre-trained weights)
        if args.pretrained and epoch == args.freeze_epochs:
            print("Unfreezing all layers...")
            set_parameter_requires_grad(model, early_layers, requires_grad=True)
            # Re-initialize optimizer to include newly unfreezed parameters
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1) # optionally lower lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)
            
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        scheduler.step()
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Checkpointing and Early Stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val Acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered!")
                break
                
    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print("\n--- Training Summary ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"History saved to: {history_path}")

    # Auto-export to ONNX if requested
    if args.export_onnx and os.path.exists(best_model_path):
        print("\n--- Exporting to ONNX ---")
        export_model(best_model_path, args.onnx_output_path)

if __name__ == "__main__":
    main()
