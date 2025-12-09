"""
Visualization module
Handles plotting and visual analytics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


class Visualizer:
    """Handles data visualization and plotting"""
    
    @staticmethod
    def plot_rating_distribution(df, title='Rating Distribution'):
        """
        Plot distribution of ratings
        
        WHAT IT DOES (Simple Explanation):
        - Creates a bar chart showing how many ratings of each type (1-5 stars)
        - Visualizes the pattern of ratings
        - Helps us understand user behavior
        
        EXAMPLE OF WHAT YOU'LL SEE:
        - X-axis: Rating values (1, 2, 3, 4, 5 stars)
        - Y-axis: Count (how many times each rating appears)
        - Typical result: 
          * 1 star: 2M ratings (users rarely give 1 star)
          * 2 stars: 5M ratings
          * 3 stars: 15M ratings
          * 4 stars: 35M ratings (most common!)
          * 5 stars: 20M ratings
        
        WHAT IT TELLS US:
        - People tend to rate movies they like (more 4s and 5s)
        - People avoid rating movies they hate (fewer 1s)
        - Distribution is "left-skewed" (more high ratings)
        - This affects our predictions!
        
        WHY IT'S USEFUL:
        - Understand data before modeling
        - Spot unusual patterns (data quality issues)
        - Know what "typical" rating looks like
        
        Args:
            df: DataFrame with rating column
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='rating', data=df, ax=ax)
        plt.title(title, fontsize=15)
        plt.ylabel('Count (millions)', fontsize=12)
        plt.xlabel('Rating', fontsize=12)
        
        # Format y-axis labels in millions
        y_labels = ax.get_yticks()
        ax.set_yticklabels([f'{int(y/1e6)}M' for y in y_labels])
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results_dict, metric='rmse', title=None):
        """
        Plot comparison of models
        
        WHAT IT DOES (Simple Explanation):
        - Creates a bar chart comparing all our models
        - Shows which model performs best (shortest bar = winner!)
        - Makes it easy to see differences at a glance
        
        EXAMPLE VISUALIZATION:
        
        |           Baseline |████████████ 1.05
        |       KNN User-User|███████████ 1.02
        |       KNN Item-Item|██████████ 1.01
        |                SVD |█████████ 0.98
        |              SVD++ |████████ 0.96
        |            XGBoost |███████ 0.95
        |  XGBoost Ensemble  |██████ 0.93  ← WINNER!
        
        HOW TO READ IT:
        - Each bar = one model
        - Bar height = error (RMSE or MAPE)
        - SHORTER bar = BETTER model (less error)
        - Numbers on bars = exact error values
        
        WHAT YOU LEARN:
        - Which model is best overall
        - How much better is it than others
        - Are simple models "good enough"?
        - Is complex model worth the extra computation?
        """
        plt.figure(figsize=(12, 6))
        
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        bars = plt.barh(models, values, color='steelblue')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{values[i]:.4f}', 
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        
        if title is None:
            title = f'Model Comparison - {metric.upper()}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_train_test_comparison(train_results, test_results, metric='rmse'):
        """
        Plot train vs test performance
        
        WHAT IT DOES (Simple Explanation):
        - Shows how models perform on TRAIN data vs TEST data side-by-side
        - Helps us detect "overfitting" (memorizing instead of learning)
        - Two bars per model: blue (train) and orange (test)
        
        EXAMPLE VISUALIZATION:
        
        Model      Train  Test
        Baseline   [■]    [■]     ← Similar heights = Good!
        SVD        [■]    [■■]    ← Test higher than train = OK
        XGBoost    [■]    [■■■]   ← Much higher test = Overfitting!
        
        WHAT IS OVERFITTING?
        - Model memorizes training data instead of learning patterns
        - Like student who memorizes answers but doesn't understand
        - Does great on practice problems (train)
        - Fails on real exam (test)
        
        HOW TO SPOT OVERFITTING:
        - Train error MUCH lower than test error
        - Example: Train RMSE = 0.5, Test RMSE = 1.2
        - Gap between blue and orange bars is huge
        
        GOOD MODEL CHARACTERISTICS:
        - Small gap between train and test error
        - Test error slightly higher is OK (expected)
        - Both errors reasonably low
        
        WHAT TO DO IF OVERFITTING:
        - Use simpler model
        - Add regularization (penalty for complexity)
        - Get more training data
        - Use cross-validation
        
        Args:
            train_results: Dictionary of train results
            test_results: Dictionary of test results
            metric: Metric to plot ('rmse' or 'mape')
        """
        models = list(train_results.keys())
        train_values = [train_results[m][metric] for m in models]
        test_values = [test_results[m][metric] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', 
                       color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', 
                       color='coral', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Train vs Test {metric.upper()}', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance', fontsize=15)
        plt.barh(range(top_n), importances[indices], color='teal', alpha=0.7)
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_ratings_over_time(df, resample_freq='M'):
        """
        Plot number of ratings over time
        
        Args:
            df: DataFrame with date and rating columns
            resample_freq: Resampling frequency ('M' for month, 'W' for week)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rating_counts = df.resample(resample_freq, on='date')['rating'].count()
        rating_counts.plot(ax=ax, linewidth=2, color='steelblue')
        
        ax.set_title('Number of Ratings Over Time', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Ratings', fontsize=12)
        
        # Format y-axis in millions
        y_labels = ax.get_yticks()
        ax.set_yticklabels([f'{int(y/1e6)}M' for y in y_labels])
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_similarity_distribution(similarity_matrix, item_id, title=None):
        """
        Plot similarity distribution for a specific item
        
        Args:
            similarity_matrix: Similarity matrix
            item_id: ID of the item
            title: Custom title
        """
        similarities = similarity_matrix[item_id].toarray().ravel()
        sim_indices = similarities.argsort()[::-1][1:]  # Exclude self
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(similarities[sim_indices], label='All similarities', 
               linewidth=2, alpha=0.7)
        ax.plot(similarities[sim_indices[:100]], label='Top 100', 
               linewidth=2, color='red')
        
        ax.set_title(title or f'Similarity Distribution for Item {item_id}', 
                    fontsize=15)
        ax.set_xlabel('Items (sorted by similarity)', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_results_table(train_results, test_results, filename='model_results.csv'):
        """
        Save model results to CSV file
        
        Args:
            train_results: Dictionary of train results
            test_results: Dictionary of test results
            filename: Output filename
        """
        # Combine results
        combined = pd.DataFrame({
            'Model': list(test_results.keys()),
            'Train_RMSE': [train_results[m]['rmse'] for m in test_results.keys()],
            'Test_RMSE': [test_results[m]['rmse'] for m in test_results.keys()],
            'Train_MAPE': [train_results[m]['mape'] for m in test_results.keys()],
            'Test_MAPE': [test_results[m]['mape'] for m in test_results.keys()]
        })
        
        # Sort by Test_RMSE
        combined = combined.sort_values('Test_RMSE')
        combined.to_csv(filename, index=False)
        
        print(f"\nResults saved to {filename}")
        print("\nModel Performance Summary:")
        print("="*80)
        print(combined.to_string(index=False))
        print("="*80 + "\n")
