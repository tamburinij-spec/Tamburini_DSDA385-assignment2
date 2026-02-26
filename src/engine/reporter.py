"""Report generation utilities for creating assignment report content."""

import json
from pathlib import Path
from datetime import datetime
import csv


def generate_metrics_table(metrics_dict, output_path=None):
    """Generate a formatted metrics table.
    
    Args:
        metrics_dict: Dict of metrics from evaluation
        output_path: Optional path to save as CSV
    
    Returns:
        Formatted string table
    """
    table = "Evaluation Metrics\n"
    table += "-" * 40 + "\n"
    table += f"{'Metric':<20} {'Value':<15}\n"
    table += "-" * 40 + "\n"
    
    for metric, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                table += f"{metric:<20} {value:<15.4f}\n"
            else:
                table += f"{metric:<20} {value:<15}\n"
    
    table += "-" * 40 + "\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
    
    return table


def generate_training_summary(summary_dict, output_path=None):
    """Generate a training summary report.
    
    Args:
        summary_dict: Summary dict from training
        output_path: Optional path to save
    
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("TRAINING AND EVALUATION SUMMARY")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Configuration
    report.append("TRAINING CONFIGURATION")
    report.append("-" * 60)
    if 'training_config' in summary_dict:
        for key, value in summary_dict['training_config'].items():
            report.append(f"  {key}: {value}")
    report.append("")
    
    # Best metrics from training
    report.append("TRAINING RESULTS")
    report.append("-" * 60)
    if 'training_metrics' in summary_dict and summary_dict['training_metrics']:
        metrics = summary_dict['training_metrics']
        final_epoch = metrics[-1]
        report.append(f"  Final Epoch: {final_epoch.get('epoch', 'N/A')}")
        report.append(f"  Final Train Loss: {final_epoch.get('train_loss', 'N/A'):.4f}")
        report.append(f"  Final Val Loss: {final_epoch.get('val_loss', 'N/A'):.4f}")
        
        # Find best validation loss
        best_val = min(metrics, key=lambda x: x['val_loss'])
        report.append(f"  Best Val Loss: {best_val['val_loss']:.4f} (Epoch {best_val['epoch']})")
    report.append("")
    
    # Test metrics
    report.append("TEST EVALUATION METRICS")
    report.append("-" * 60)
    if 'test_metrics' in summary_dict:
        for key, value in summary_dict['test_metrics'].items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
    report.append("")
    
    report.append("=" * 60)
    
    full_report = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
    
    return full_report


def save_training_log_csv(metrics_history, output_path):
    """Save training metrics to CSV file.
    
    Args:
        metrics_history: List of metric dicts from each epoch
        output_path: Path to save CSV
    """
    if not metrics_history:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
        writer.writeheader()
        writer.writerows(metrics_history)
    
    print(f"✓ Training log saved to {output_path}")


def generate_comparison_table(model1_metrics, model2_metrics=None, output_path=None):
    """Generate a comparison table for multiple models.
    
    Args:
        model1_metrics: Dict of metrics for first model
        model2_metrics: Dict of metrics for second model (optional)
        output_path: Optional path to save as CSV
    
    Returns:
        Formatted string table
    """
    table = "Model Comparison\n"
    table += "-" * 60 + "\n"
    
    # Header
    header = f"{'Metric':<20}"
    if model1_metrics:
        header += f" {'Model 1':<15}"
    if model2_metrics:
        header += f" {'Model 2':<15}"
    table += header + "\n"
    table += "-" * 60 + "\n"
    
    # Get all unique metrics
    all_metrics = set()
    if model1_metrics:
        all_metrics.update(model1_metrics.keys())
    if model2_metrics:
        all_metrics.update(model2_metrics.keys())
    
    # Add rows
    for metric in sorted(all_metrics):
        row = f"{metric:<20}"
        if model1_metrics and metric in model1_metrics:
            val = model1_metrics[metric]
            if isinstance(val, float):
                row += f" {val:<15.4f}"
            else:
                row += f" {str(val):<15}"
        elif model1_metrics:
            row += f" {'N/A':<15}"
        
        if model2_metrics and metric in model2_metrics:
            val = model2_metrics[metric]
            if isinstance(val, float):
                row += f" {val:<15.4f}"
            else:
                row += f" {str(val):<15}"
        elif model2_metrics:
            row += f" {'N/A':<15}"
        
        table += row + "\n"
    
    table += "-" * 60 + "\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
    
    return table


def create_experiment_report(results_dir='outputs'):
    """Create a comprehensive experiment report.
    
    Args:
        results_dir: Directory containing results
    
    Returns:
        Report string
    """
    results_dir = Path(results_dir)
    summary_path = results_dir / 'predictions' / 'summary.json'
    
    if not summary_path.exists():
        return "Summary file not found"
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    report = generate_training_summary(summary)
    
    # Save report
    report_path = results_dir / 'predictions' / 'REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to {report_path}")
    return report
