import sys
from pathlib import Path

# ensure project root is on path when running from the src/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import json

from config import DEVICE, TRAINING_CONFIG, MODEL_CONFIG

from datasets.pennfudan import get_data_loaders
from engine.train import Trainer
from engine.inference import run_test_evaluation
from engine.visualization import save_predictions_visualization
from engine.reporter import generate_training_summary, save_training_log_csv
from models.faster_rcnn import create_faster_rcnn


def main():
    print(f"Using device: {DEVICE}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=int(TRAINING_CONFIG['batch_size'])
    )
    
    # Create detection model (Faster R-CNN)
    print("Creating Faster R-CNN model...")
    model = create_faster_rcnn(
        num_classes=MODEL_CONFIG['num_classes'],
        device=DEVICE
    )
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(TRAINING_CONFIG['learning_rate']),
        weight_decay=float(TRAINING_CONFIG['weight_decay'])
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Create trainer with checkpoint directory
    checkpoint_dir = Path('outputs/checkpoints')
    print(f"\n[DEBUG] Main: checkpoint_dir = {checkpoint_dir.absolute()}")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=int(TRAINING_CONFIG['num_epochs']),
        checkpoint_dir=checkpoint_dir
    )
    
    # Train model
    print("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
        print("[DEBUG] Training completed successfully")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load best model for evaluation and testing
    print("\nLoading best model for evaluation...")
    try:
        best_model_path = checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from {best_model_path}")
        else:
            print(f"⚠ Best model not found at {best_model_path}, using current model")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    # Run test set evaluation
    print("\nEvaluating on test set...")
    try:
        output_dir = Path('outputs/predictions')
        output_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = run_test_evaluation(model, test_loader, DEVICE, output_dir)
        print("[DEBUG] Test evaluation completed")
    except Exception as e:
        print(f"[ERROR] Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        test_metrics = {}
    
    # Generate prediction visualizations
    print("\nGenerating prediction visualizations...")
    try:
        save_predictions_visualization(model, test_loader, DEVICE, output_dir)
        print("[DEBUG] Visualization completed")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save summary report
    print("\nGenerating summary report...")
    try:
        summary = {
            'training_config': dict(TRAINING_CONFIG),
            'model_config': dict(MODEL_CONFIG),
            'device': str(DEVICE),
            'best_model_path': str(best_model_path),
            'training_metrics': trainer.metrics_history,
            'test_metrics': test_metrics,
            'output_directory': str(output_dir)
        }
        
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary report saved to {summary_path}")
    except Exception as e:
        print(f"[ERROR] Summary save failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save training log as CSV
    try:
        log_csv_path = output_dir / 'training_log.csv'
        save_training_log_csv(trainer.metrics_history, log_csv_path)
    except Exception as e:
        print(f"[ERROR] CSV save failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate text report
    try:
        report_text = generate_training_summary(summary)
        report_path = output_dir / 'REPORT.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Training report saved to {report_path}")
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*60)
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {output_dir}")
    print(f"\nTest Metrics:")
    print(f"  Precision: {test_metrics['Precision']:.4f}")
    print(f"  Recall: {test_metrics['Recall']:.4f}")
    print(f"  F1-Score: {test_metrics['F1-Score']:.4f}")
    print(f"  Mean IoU: {test_metrics['Mean_IoU']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()