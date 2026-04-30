import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2
from scipy.stats import pearsonr
from collections import Counter

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Target categories
TARGET_CATEGORIES = [
    'Sports',
    'Gaming',
    'Music',
    'Entertainment',
    'Film & Animation',
    'Politics',
    'People & Blogs',
    'Education',
    'Science & Technology'
]


def analyze_thumbnail_quality(image_path):
    """
    Analyze thumbnail quality metrics.
    Returns: dict with brightness, contrast, resolution, sharpness, colorfulness
    """
    try:
        # Open image with PIL
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Resolution (width x height)
        width, height = img.size
        resolution = width * height
        
        # Brightness: average pixel value (0-255)
        brightness = np.mean(img_array)
        
        # Contrast: standard deviation of pixel values
        contrast = np.std(img_array)
        
        # Colorfulness: based on research paper formula
        rg = img_array[:, :, 0].astype(float) - img_array[:, :, 1].astype(float)
        yb = 0.5 * (img_array[:, :, 0].astype(float) + img_array[:, :, 1].astype(float)) - img_array[:, :, 2].astype(float)
        colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        
        # Sharpness: Laplacian variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Saturation: HSV analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'resolution': float(resolution),
            'sharpness': float(sharpness),
            'colorfulness': float(colorfulness),
            'saturation': float(saturation),
            'width': width,
            'height': height
        }
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None


def load_and_prepare_data(csv_path, thumbnails_dir):
    """
    Load CSV metadata and match with downloaded thumbnails.
    Returns: DataFrame with metadata + quality metrics
    """
    print("Loading video metadata from CSV...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Normalize category names
    df['category'] = df['category'].str.strip()
    
    # Filter for target categories
    df = df[df['category'].isin(TARGET_CATEGORIES)].copy()
    print(f"Filtered to {len(df)} videos in target categories")
    
    # Convert engagement metrics
    for col in ['views', 'likes', 'comments']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing engagement data
    df = df.dropna(subset=['views', 'likes', 'comments'])
    
    # Analyze thumbnails
    print("Analyzing thumbnail quality...")
    quality_data = []
    
    for idx, row in df.iterrows():
        video_id = row['video_id']
        thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
        
        if os.path.exists(thumbnail_path):
            quality = analyze_thumbnail_quality(thumbnail_path)
            if quality:
                quality_data.append({
                    'video_id': video_id,
                    'quality': quality
                })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} videos...")
    
    # Merge quality data back
    quality_df = pd.DataFrame(quality_data)
    if quality_df.empty:
        raise ValueError("No thumbnail quality data extracted")
    
    # Expand quality dict into separate columns
    quality_expanded = pd.json_normalize(quality_df['quality'])
    quality_df = pd.concat([quality_df[['video_id']], quality_expanded], axis=1)
    
    # Merge with original metadata
    df = df.merge(quality_df, on='video_id', how='inner')
    
    print(f"Successfully processed {len(df)} videos with thumbnails")
    return df


def calculate_correlations(df):
    """
    Calculate correlations between thumbnail quality and engagement metrics per category.
    """
    quality_metrics = ['brightness', 'contrast', 'resolution', 'sharpness', 'colorfulness', 'saturation']
    engagement_metrics = ['views', 'likes', 'comments']
    
    correlations = {}
    
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if len(cat_df) < 3:  # Need minimum samples
            continue
        
        cat_corr = {}
        for quality_metric in quality_metrics:
            for engagement_metric in engagement_metrics:
                try:
                    # Use log transformation for engagement metrics to handle skew
                    engagement_log = np.log1p(cat_df[engagement_metric])
                    corr, p_value = pearsonr(cat_df[quality_metric], engagement_log)
                    cat_corr[f"{quality_metric}_vs_{engagement_metric}"] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(cat_df)
                    }
                except Exception as e:
                    print(f"Error calculating correlation for {category}: {e}")
        
        correlations[category] = cat_corr
    
    return correlations


def generate_correlation_heatmap(df, output_dir):
    """Generate correlation heatmaps per category."""
    
    quality_metrics = ['brightness', 'contrast', 'resolution', 'sharpness', 'colorfulness', 'saturation']
    engagement_metrics = ['views', 'likes', 'comments']
    
    for category in TARGET_CATEGORIES:
        cat_df = df[df['category'] == category]
        if len(cat_df) < 3:
            print(f"Skipping {category}: insufficient data")
            continue
        
        # Create correlation matrix
        metrics_to_use = quality_metrics + engagement_metrics
        available_cols = [col for col in metrics_to_use if col in cat_df.columns]
        
        # Apply log transformation to engagement metrics
        cat_df_transformed = cat_df[available_cols].copy()
        for metric in engagement_metrics:
            if metric in cat_df_transformed.columns:
                cat_df_transformed[metric] = np.log1p(cat_df_transformed[metric])
        
        corr_matrix = cat_df_transformed.corr()
        
        # Filter to show quality vs engagement correlations
        quality_engagement_corr = corr_matrix.loc[quality_metrics, engagement_metrics]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(quality_engagement_corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, cbar_kws={'label': 'Pearson Correlation'})
        plt.title(f'{category}: Thumbnail Quality vs Engagement Metrics')
        plt.xlabel('Engagement Metrics (log-transformed)')
        plt.ylabel('Thumbnail Quality Metrics')
        plt.tight_layout()
        
        filename = f"{category.replace(' ', '_').replace('&', 'and')}_correlation_heatmap.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated: {filename}")


def generate_scatter_plots(df, output_dir):
    """Generate scatter plots for key relationships per category."""
    
    for category in TARGET_CATEGORIES:
        cat_df = df[df['category'] == category]
        if len(cat_df) < 3:
            continue
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{category}: Thumbnail Quality vs Video Engagement', fontsize=16, fontweight='bold')
        
        quality_metrics = ['brightness', 'contrast', 'resolution', 'sharpness', 'colorfulness', 'saturation']
        
        for idx, quality_metric in enumerate(quality_metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Plot against views (primary engagement metric)
            ax.scatter(cat_df[quality_metric], np.log1p(cat_df['views']), alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(cat_df[quality_metric], np.log1p(cat_df['views']), 1)
            p = np.poly1d(z)
            x_line = np.linspace(cat_df[quality_metric].min(), cat_df[quality_metric].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr, _ = pearsonr(cat_df[quality_metric], np.log1p(cat_df['views']))
            
            ax.set_xlabel(quality_metric.capitalize())
            ax.set_ylabel('Views (log scale)')
            ax.set_title(f'{quality_metric} vs Views (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{category.replace(' ', '_').replace('&', 'and')}_scatter_plots.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated: {filename}")


def generate_category_comparison_chart(df, output_dir):
    """Generate a comparison chart of average metrics across categories."""
    
    quality_metrics = ['brightness', 'contrast', 'sharpness', 'colorfulness', 'saturation']
    
    category_stats = []
    for category in TARGET_CATEGORIES:
        cat_df = df[df['category'] == category]
        if len(cat_df) < 3:
            continue
        
        stats = {
            'category': category,
            'avg_views': cat_df['views'].mean(),
            'avg_likes': cat_df['likes'].mean(),
            'avg_engagement_rate': (cat_df['likes'] / cat_df['views'] * 100).mean(),
        }
        
        for metric in quality_metrics:
            stats[f'avg_{metric}'] = cat_df[metric].mean()
        
        category_stats.append(stats)
    
    stats_df = pd.DataFrame(category_stats)
    
    # Plot average quality metrics by category
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Average Thumbnail Quality Metrics by Category', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(quality_metrics):
        ax = axes[idx // 3, idx % 3]
        
        x = range(len(stats_df))
        y = stats_df[f'avg_{metric}']
        
        bars = ax.bar(x, y, color='steelblue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['category'], rotation=45, ha='right')
        ax.set_ylabel(f'Average {metric.capitalize()}')
        ax.set_title(metric.capitalize())
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: category_comparison.png")


def generate_engagement_distribution(df, output_dir):
    """Generate engagement distribution by category."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Engagement Metrics Distribution by Category', fontsize=16, fontweight='bold')
    
    engagement_metrics = ['views', 'likes', 'comments']
    
    for idx, metric in enumerate(engagement_metrics):
        ax = axes[idx]
        
        data_by_cat = [df[df['category'] == cat][metric].values for cat in TARGET_CATEGORIES 
                       if len(df[df['category'] == cat]) > 0]
        
        ax.boxplot(data_by_cat, labels=[cat[:10] for cat in TARGET_CATEGORIES if len(df[df['category'] == cat]) > 0])
        ax.set_ylabel(f'{metric.capitalize()} (log scale)')
        ax.set_yscale('log')
        ax.set_title(f'{metric.capitalize()} Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'engagement_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated: engagement_distribution.png")


def save_correlation_report(correlations, output_dir):
    """Save detailed correlation report as JSON."""
    
    report = {}
    for category, cat_corr in correlations.items():
        report[category] = {}
        for key, value in cat_corr.items():
            report[category][key] = {
                'correlation': round(value['correlation'], 4),
                'p_value': round(value['p_value'], 6),
                'n_samples': value['n_samples']
            }
    
    report_path = os.path.join(output_dir, 'correlation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved correlation report: {report_path}")


def main():
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "US_Trending_filtered.csv"
    thumbnails_dir = project_root / "data" / "thumbnails_US_Trending"
    output_dir = project_root / "data" / "analysis" / "thumbnail_quality_analysis"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreated output directory if missing.")
    
    # Load and prepare data
    print("\nStarting data preparation...")
    df = load_and_prepare_data(str(csv_path), str(thumbnails_dir))
    print(f"Data preparation complete. {len(df)} videos with valid thumbnails will be analyzed.")
    
    print(f"\nAnalyzing {len(df)} videos across categories:")
    print(df['category'].value_counts())
    
    # Calculate correlations
    print("\nCalculating correlations...")
    correlations = calculate_correlations(df)
    print("Correlation calculation complete.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_correlation_heatmap(df, str(output_dir))
    print("Correlation heatmaps generated.")
    
    generate_scatter_plots(df, str(output_dir))
    print("Scatter plots generated.")
    
    generate_category_comparison_chart(df, str(output_dir))
    print("Category comparison chart generated.")
    
    generate_engagement_distribution(df, str(output_dir))
    print("Engagement distribution chart generated.")
    
    # Save report
    print("\nSaving correlation report...")
    save_correlation_report(correlations, str(output_dir))
    print("Correlation report saved.")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    for category in sorted(correlations.keys()):
        print(f"\n{category}:")
        cat_corr = correlations[category]
        
        # Find strongest correlations
        strongest = sorted([(k, v['correlation']) for k, v in cat_corr.items()],
                          key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for key, corr in strongest:
            print(f"  {key}: {corr:.4f}")
    
    print(f"\n✓ All analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
