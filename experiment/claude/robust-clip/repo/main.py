#!/usr/bin/env python
"""
Main entrypoint for Robust CLIP reproduction.

This script orchestrates FARE adversarial fine-tuning, robustness evaluation,
and artifact generation for the paper "Robust CLIP: Unsupervised Adversarial
Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models".

Usage:
    python main.py --mode train --model fare --epsilon 4/255
    python main.py --mode eval --model fare --dataset imagenet
    python main.py --mode runtime_smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Robust CLIP: FARE adversarial fine-tuning and evaluation'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='runtime_smoke',
        choices=['train', 'eval', 'train_eval', 'runtime_smoke', 'docker_validate'],
        help='Execution mode: train, eval, train_eval, runtime_smoke, or docker_validate'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='fare',
        choices=['clip', 'tecoa', 'fare'],
        help='Model variant to train or evaluate'
    )
    
    parser.add_argument(
        '--epsilon',
        type=str,
        default='4/255',
        help='Adversarial perturbation budget (e.g., 4/255, 8/255, 16/255)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=['imagenet'],
        help='Dataset(s) for evaluation (e.g., imagenet, cifar10, cifar100)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for results and checkpoints'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation or resuming training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for computation (cuda or cpu)'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {
            'training': {
                'num_epochs': 10,
                'batch_size': 128,
                'learning_rate': 1e-5,
                'epsilon_values': [4/255, 8/255, 16/255]
            },
            'evaluation': {
                'datasets': ['imagenet', 'cifar10', 'cifar100'],
                'epsilon_values': [4/255, 8/255, 16/255]
            },
            'model': {
                'architecture': 'ViT-L/14',
                'pretrained': 'openai'
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_artifact_directories(output_dir: str):
    """Ensure all required output directories exist."""
    directories = [
        os.path.join(output_dir, 'figures'),
        os.path.join(output_dir, 'results'),
        os.path.join(output_dir, 'checkpoints')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories


def run_training(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Execute training pipeline for the specified model variant."""
    from src.training import train_robust_clip
    from src.data import get_training_dataloader
    from src.models import load_clip_model
    
    print(f"Starting {args.model.upper()} training with epsilon={args.epsilon}")
    
    # Load model
    model = load_clip_model(
        architecture=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        device=args.device
    )
    
    # Get training data
    train_loader = get_training_dataloader(
        dataset_name='imagenet',
        batch_size=args.batch_size,
        split='train',
        config=config
    )
    
    # Parse epsilon
    epsilon = eval(args.epsilon) if '/' in args.epsilon else float(args.epsilon)
    
    # Train model
    training_results = train_robust_clip(
        model=model,
        train_loader=train_loader,
        method=args.model,
        epsilon=epsilon,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir,
        config=config
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'{args.model}_clip.pth')
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'args': vars(args),
        'training_results': training_results
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    final_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'{args.model}_clip_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'args': vars(args),
        'training_results': training_results,
        'final': True
    }, final_checkpoint_path)
    print(f"Final checkpoint saved to {final_checkpoint_path}")
    
    return training_results


def run_evaluation(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Execute evaluation pipeline for the specified model variant."""
    from src.evaluation import evaluate_robustness
    from src.data import get_evaluation_dataloader
    from src.models import load_clip_model
    
    print(f"Starting {args.model.upper()} evaluation on {args.dataset}")
    
    # Load model
    if args.checkpoint:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model = load_clip_model(
            architecture=config['model']['architecture'],
            pretrained=config['model']['pretrained'],
            device=args.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        model = load_clip_model(
            architecture=config['model']['architecture'],
            pretrained=config['model']['pretrained'],
            device=args.device
        )
    
    # Parse epsilon
    epsilon = eval(args.epsilon) if '/' in args.epsilon else float(args.epsilon)
    
    # Evaluate on each dataset
    all_results = {}
    for dataset_name in args.dataset:
        print(f"\nEvaluating on {dataset_name}...")
        
        eval_loader = get_evaluation_dataloader(
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            split='test',
            config=config
        )
        
        results = evaluate_robustness(
            model=model,
            eval_loader=eval_loader,
            dataset_name=dataset_name,
            epsilon=epsilon,
            device=args.device,
            config=config
        )
        
        all_results[dataset_name] = results
        print(f"{dataset_name} results: Clean Acc={results['clean_accuracy']:.2f}%, "
              f"Adv Acc={results['adversarial_accuracy']:.2f}%")
    
    return all_results


def write_artifacts(
    training_results: Optional[Dict[str, Any]],
    evaluation_results: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    args: argparse.Namespace
):
    """Write all output artifacts including figures, metrics, and tables."""
    from src.artifacts import write_metrics_json, write_robustness_table
    from src.plotting import create_radar_plot
    
    output_dir = args.output_dir
    
    # Write metrics JSON
    if evaluation_results:
        metrics_path = os.path.join(output_dir, 'results', 'metrics.json')
        metrics_data = {
            'model': args.model,
            'epsilon': args.epsilon,
            'evaluation_results': evaluation_results,
            'config': config
        }
        if training_results:
            metrics_data['training_results'] = training_results
        
        write_metrics_json(metrics_data, metrics_path)
        print(f"Metrics written to {metrics_path}")
    
    # Write robustness table
    if evaluation_results:
        table_path = os.path.join(output_dir, 'results', 'robustness_table.csv')
        write_robustness_table(evaluation_results, table_path, args.model)
        print(f"Robustness table written to {table_path}")
    
    # Create radar plot
    if evaluation_results:
        radar_png_path = os.path.join(output_dir, 'figures', 'figure1_radar.png')
        radar_pdf_path = os.path.join(output_dir, 'figures', 'figure1_radar.pdf')
        create_radar_plot(evaluation_results, radar_png_path, radar_pdf_path, args.model)
        print(f"Radar plots written to {radar_png_path} and {radar_pdf_path}")


def create_smoke_artifacts(output_dir: str):
    """Create minimal schema artifacts for smoke testing."""
    ensure_artifact_directories(output_dir)
    
    # Create schema for metrics.json
    metrics_path = os.path.join(output_dir, 'results', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            '_artifact_type': 'dry_run_schema',
            '_note': 'This is a contract validation artifact, not real results',
            'model': 'fare',
            'epsilon': '4/255',
            'evaluation_results': {
                'imagenet': {
                    'clean_accuracy': 0.0,
                    'adversarial_accuracy': 0.0
                }
            }
        }, f, indent=2)
    
    # Create schema for robustness_table.csv
    table_path = os.path.join(output_dir, 'results', 'robustness_table.csv')
    with open(table_path, 'w') as f:
        f.write('_artifact_type,dry_run_schema\n')
        f.write('dataset,model,clean_acc,adv_acc\n')
        f.write('imagenet,fare,0.0,0.0\n')
    
    # Create minimal diagnostic radar plots
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        ax.text(0.5, 0.5, 'Dry-run schema artifact\nNot real results',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        radar_png_path = os.path.join(output_dir, 'figures', 'figure1_radar.png')
        plt.savefig(radar_png_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        radar_pdf_path = os.path.join(output_dir, 'figures', 'figure1_radar.pdf')
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        ax.text(0.5, 0.5, 'Dry-run schema artifact\nNot real results',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.savefig(radar_pdf_path, bbox_inches='tight')
        plt.close()
    except ImportError:
        # Fallback if matplotlib not available
        for path in ['figures/figure1_radar.png', 'figures/figure1_radar.pdf']:
            full_path = os.path.join(output_dir, path)
            with open(full_path, 'w') as f:
                f.write('# Dry-run schema artifact placeholder\n')
    
    # Create checkpoint schemas
    checkpoint_path = os.path.join(output_dir, 'checkpoints', 'fare_clip.pth')
    with open(checkpoint_path, 'w') as f:
        f.write('# Dry-run checkpoint schema\n')
    
    final_checkpoint_path = os.path.join(output_dir, 'checkpoints', 'fare_clip_final.pth')
    with open(final_checkpoint_path, 'w') as f:
        f.write('# Dry-run final checkpoint schema\n')
    
    # Create readiness.json
    readiness_path = os.path.join(output_dir, 'readiness.json')
    with open(readiness_path, 'w') as f:
        json.dump({
            '_artifact_type': 'readiness_manifest',
            '_note': 'Contract validation only, not experiment results',
            'status': 'artifacts_created',
            'mode': 'smoke_validation',
            'artifacts': [
                'figures/figure1_radar.png',
                'figures/figure1_radar.pdf',
                'results/metrics.json',
                'results/robustness_table.csv',
                'checkpoints/fare_clip.pth',
                'checkpoints/fare_clip_final.pth'
            ]
        }, f, indent=2)
    
    # Create evaluation_result.json
    eval_result_path = os.path.join(output_dir, 'evaluation_result.json')
    with open(eval_result_path, 'w') as f:
        json.dump({
            '_artifact_type': 'evaluation_schema',
            '_note': 'Contract validation only, not real evaluation',
            'status': 'schema_validated',
            'model': 'fare',
            'datasets': ['imagenet'],
            'metrics': {
                'clean_accuracy': 0.0,
                'adversarial_accuracy': 0.0
            }
        }, f, indent=2)
    
    print("Smoke validation artifacts created successfully")


def main():
    """Main execution function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_epochs:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    # Ensure output directories exist
    ensure_artifact_directories(args.output_dir)
    
    # Handle smoke/validation modes
    if args.mode in ['runtime_smoke', 'docker_validate']:
        print(f"Running {args.mode} validation...")
        create_smoke_artifacts(args.output_dir)
        print(f"{args.mode} completed successfully")
        return
    
    # Execute training and/or evaluation
    training_results = None
    evaluation_results = None
    
    if args.mode in ['train', 'train_eval']:
        training_results = run_training(config, args)
    
    if args.mode in ['eval', 'train_eval']:
        evaluation_results = run_evaluation(config, args)
    
    # Write all artifacts
    if training_results or evaluation_results:
        write_artifacts(training_results, evaluation_results, config, args)
    
    print("Execution completed successfully")


if __name__ == '__main__':
    main()