import json
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json_results(directory: str) -> dict[str, dict]:
    results = {}
    json_files = list(Path(directory).glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            model_name = data['metadata']['model_name']
            results[model_name] = data
    
    return results

def calculate_average_tokens(result_data: dict) -> float:
    tokens = [r['generated_tokens'] for r in result_data['results'] if r['generated_tokens'] is not None]
    return sum(tokens) / len(tokens) if tokens else 0

def plot_accuracy_vs_tokens(results: dict[str, dict], output_dir: str):
    models = []
    accuracies = []
    avg_tokens = []
    
    for model_name, data in results.items():
        models.append(model_name)
        accuracies.append(data['metadata']['stats']['problem_accuracy'])
        avg_tokens.append(calculate_average_tokens(data))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_tokens, accuracies, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(models):
        plt.annotate(model, (avg_tokens[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    plt.xlabel('Average Tokens Generated', fontsize=12)
    plt.ylabel('Problem Accuracy (%)', fontsize=12)
    plt.title('Model Performance vs Token Generation', fontsize=14)
    plt.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accuracy_vs_tokens.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs tokens plot to {output_path}")

def plot_accuracy_bars(results: dict[str, dict], output_dir: str):
    # Sort models by problem accuracy (best to worst)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['metadata']['stats']['problem_accuracy'], reverse=True)
    
    models = [m[0] for m in sorted_models]
    problem_acc = [m[1]['metadata']['stats']['problem_accuracy'] for m in sorted_models]
    attempt_acc = [m[1]['metadata']['stats']['attempt_accuracy'] for m in sorted_models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, problem_acc, width, label='Problem Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, attempt_acc, width, label='Attempt Accuracy', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison plot to {output_path}")

def plot_heatmap(results: dict[str, dict], output_dir: str):
    model_problem_acc = {}
    all_problem_ids = set()
    
    for model_name, data in results.items():
        problem_acc = {}
        for result in data['results']:
            problem_id = result['problem_id']
            all_problem_ids.add(problem_id)
            if problem_id not in problem_acc:
                problem_acc[problem_id] = {'correct': 0, 'total': 0}
            problem_acc[problem_id]['total'] += 1
            if result['result_type'] == 'correct':
                problem_acc[problem_id]['correct'] += 1
        
        model_problem_acc[model_name] = {
            pid: (acc['correct'] / acc['total'] * 100) if acc['total'] > 0 else 0
            for pid, acc in problem_acc.items()
        }
    
    problem_difficulties = {}
    for pid in all_problem_ids:
        accs = [model_problem_acc[model].get(pid, 0) for model in model_problem_acc]
        problem_difficulties[pid] = sum(accs) / len(accs)
    
    sorted_problems = sorted(all_problem_ids, key=lambda x: problem_difficulties[x], reverse=True)
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['metadata']['stats']['problem_accuracy'], reverse=True)
    
    matrix = []
    for model in sorted_models:
        row = [model_problem_acc[model].get(pid, 0) for pid in sorted_problems]
        matrix.append(row)
    
    # Calculate figure size to ensure square cells
    n_models = len(sorted_models)
    n_problems = len(sorted_problems)
    cell_size = 0.5
    fig_width = n_problems * cell_size + 3
    fig_height = n_models * cell_size + 2
    
    # Create heatmap
    plt.figure(figsize=(fig_width, fig_height))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    ax = sns.heatmap(
        matrix, 
        xticklabels=[f'P{pid}' for pid in sorted_problems],
        yticklabels=sorted_models,
        cmap=cmap,
        vmin=0, vmax=100,
        annot=True,
        fmt='.0f',
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    
    plt.xlabel('Problem ID (Easiest → Hardest)', fontsize=12)
    plt.ylabel('Model (Best → Worst)', fontsize=12)
    plt.title('Model Performance Heatmap by Problem', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot AIME evaluation results')
    parser.add_argument('directory', type=str, help='Directory containing JSON result files')
    parser.add_argument('-o', '--output-dir', type=str, default='plots', help='Output directory for plots (default: plots)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all results
    print(f"Loading JSON files from {args.directory}...")
    results = load_json_results(args.directory)
    print(f"Loaded results for {len(results)} models")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_vs_tokens(results, args.output_dir)
    plot_accuracy_bars(results, args.output_dir)
    plot_heatmap(results, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
