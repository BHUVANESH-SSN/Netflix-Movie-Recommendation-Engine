"""
Netflix Movie Recommendation Engine
====================================
A comprehensive machine learning system for movie recommendation using collaborative filtering,
matrix factorization, and ensemble methods.

Author: BHUVANESH SSN
Date: December 2025
"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import randint as sp_randint, uniform as stats_uniform

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from surprise import Reader, Dataset, BaselineOnly, KNNBaseline, SVD, SVDpp


# ==================== CONFIGURATION ====================
class Config:
    """Configuration constants for the recommendation system"""
    DATA_FOLDER = 'data_folder'
    COMBINED_FILES = ['combined_data_1.txt', 'combined_data_2.txt', 
                      'combined_data_3.txt', 'combined_data_4.txt']
    MOVIE_TITLES_FILE = 'movie_titles.csv'
    
    TRAIN_SPLIT = 0.80
    TEST_SPLIT = 0.20
    
    RANDOM_SEED = 15
    
    # Sampling parameters
    SAMPLE_USERS = 10000
    SAMPLE_MOVIES = 1000
    SAMPLE_TEST_USERS = 5000
    SAMPLE_TEST_MOVIES = 500


# ==================== DATA PREPROCESSING ====================
class DataPreprocessor:
    """Handles data loading, preprocessing, and transformation"""
    
    @staticmethod
    def merge_netflix_data(output_file='data.csv', data_folder='data_folder'):
        """
        Merge all Netflix rating files into a single CSV file
        
        Args:
            output_file: Output CSV filename
            data_folder: Folder containing Netflix data files
            
        Returns:
            None (saves data to disk)
        """
        if os.path.isfile(output_file):
            print(f"{output_file} already exists. Skipping merge.")
            return
        
        print("Merging Netflix data files...")
        start = datetime.now()
        
        data = open(output_file, mode='w')
        row = list()
        
        files = [f'{data_folder}/{f}' for f in Config.COMBINED_FILES]
        
        for file in files:
            print(f"Reading ratings from {file}...")
            with open(file) as f:
                for line in f:
                    del row[:]
                    line = line.strip()
                    if line.endswith(':'):
                        movie_id = line.replace(':', '')
                    else:
                        row = [x for x in line.split(',')]
                        row.insert(0, movie_id)
                        data.write(','.join(row))
                        data.write('\n')
            print("Done.\n")
        
        data.close()
        print(f'Time taken: {datetime.now() - start}')
    
    @staticmethod
    def load_and_sort_data(filename='data.csv'):
        """
        Load data from CSV and sort by date
        
        Args:
            filename: CSV file to load
            
        Returns:
            pd.DataFrame: Sorted dataframe
        """
        print(f"Loading dataframe from {filename}...")
        df = pd.read_csv(filename, sep=',', 
                        names=['movie', 'user', 'rating', 'date'])
        df.date = pd.to_datetime(df.date)
        print('Done.\n')
        
        print('Sorting dataframe by date...')
        df.sort_values(by='date', inplace=True)
        print('Done.')
        
        return df
    
    @staticmethod
    def check_data_quality(df):
        """
        Check for NaN values and duplicates
        
        Args:
            df: DataFrame to check
            
        Returns:
            dict: Quality metrics
        """
        nan_count = sum(df.isnull().any())
        dup_count = sum(df.duplicated(['movie', 'user', 'rating']))
        
        metrics = {
            'nan_values': nan_count,
            'duplicates': dup_count,
            'total_ratings': df.shape[0],
            'unique_users': len(np.unique(df.user)),
            'unique_movies': len(np.unique(df.movie))
        }
        
        print("Data Quality Metrics:")
        print("-" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        return metrics
    
    @staticmethod
    def split_train_test(df, train_ratio=0.80):
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for train split
            
        Returns:
            tuple: (train_df, test_df)
        """
        split_idx = int(df.shape[0] * train_ratio)
        
        if not os.path.isfile('train.csv'):
            df.iloc[:split_idx].to_csv("train.csv", index=False)
            print("Train data saved to train.csv")
        
        if not os.path.isfile('test.csv'):
            df.iloc[split_idx:].to_csv("test.csv", index=False)
            print("Test data saved to test.csv")
        
        train_df = pd.read_csv("train.csv", parse_dates=['date'])
        test_df = pd.read_csv("test.csv", parse_dates=['date'])
        
        return train_df, test_df


# ==================== SPARSE MATRIX OPERATIONS ====================
class SparseMatrixHandler:
    """Handles sparse matrix creation and operations"""
    
    @staticmethod
    def create_sparse_matrix(df, filename=None):
        """
        Create sparse matrix from dataframe
        
        Args:
            df: DataFrame with user, movie, rating columns
            filename: Optional filename to save/load matrix
            
        Returns:
            csr_matrix: Sparse matrix
        """
        if filename and os.path.isfile(filename):
            print(f"Loading sparse matrix from {filename}...")
            return sparse.load_npz(filename)
        
        print("Creating sparse matrix from dataframe...")
        start = datetime.now()
        
        sparse_matrix = sparse.csr_matrix(
            (df.rating.values, (df.user.values, df.movie.values))
        )
        
        print(f'Done. Shape: {sparse_matrix.shape}')
        print(f'Time taken: {datetime.now() - start}')
        
        if filename:
            print(f'Saving to {filename}...')
            sparse.save_npz(filename, sparse_matrix)
            print('Done.')
        
        return sparse_matrix
    
    @staticmethod
    def calculate_sparsity(sparse_matrix):
        """
        Calculate sparsity percentage of matrix
        
        Args:
            sparse_matrix: Sparse matrix
            
        Returns:
            float: Sparsity percentage
        """
        users, movies = sparse_matrix.shape
        elements = sparse_matrix.count_nonzero()
        sparsity = (1 - (elements / (users * movies))) * 100
        
        print(f"Sparsity: {sparsity:.2f}%")
        return sparsity
    
    @staticmethod
    def sample_sparse_matrix(sparse_matrix, no_users, no_movies, path):
        """
        Sample a subset of sparse matrix
        
        Args:
            sparse_matrix: Original sparse matrix
            no_users: Number of users to sample
            no_movies: Number of movies to sample
            path: Path to save sampled matrix
            
        Returns:
            csr_matrix: Sampled sparse matrix
        """
        if os.path.isfile(path):
            print(f"Loading sampled matrix from {path}...")
            return sparse.load_npz(path)
        
        print(f"Creating sample with {no_users} users and {no_movies} movies...")
        
        row_ind, col_ind, ratings = sparse.find(sparse_matrix)
        users = np.unique(row_ind)
        movies = np.unique(col_ind)
        
        np.random.seed(Config.RANDOM_SEED)
        sample_users = np.random.choice(users, no_users, replace=False)
        sample_movies = np.random.choice(movies, no_movies, replace=False)
        
        mask = np.logical_and(
            np.isin(row_ind, sample_users),
            np.isin(col_ind, sample_movies)
        )
        
        sample_sparse_matrix = sparse.csr_matrix(
            (ratings[mask], (row_ind[mask], col_ind[mask])),
            shape=(max(sample_users) + 1, max(sample_movies) + 1)
        )
        
        print(f'Sampled Matrix Ratings: {ratings[mask].shape[0]}')
        sparse.save_npz(path, sample_sparse_matrix)
        print('Done.')
        
        return sample_sparse_matrix


# ==================== FEATURE ENGINEERING ====================
class FeatureEngineer:
    """Handles feature engineering for ML models"""
    
    @staticmethod
    def get_average_ratings(sparse_matrix, of_users=True):
        """
        Calculate average ratings per user or per movie
        
        Args:
            sparse_matrix: Sparse rating matrix
            of_users: True for user averages, False for movie averages
            
        Returns:
            dict: Dictionary of averages
        """
        axis = 1 if of_users else 0
        
        sum_of_ratings = sparse_matrix.sum(axis=axis).A1
        is_rated = sparse_matrix != 0
        no_of_ratings = is_rated.sum(axis=axis).A1
        
        u, m = sparse_matrix.shape
        average_ratings = {
            i: sum_of_ratings[i] / no_of_ratings[i]
            for i in range(u if of_users else m)
            if no_of_ratings[i] != 0
        }
        
        return average_ratings
    
    @staticmethod
    def compute_global_average(sparse_matrix):
        """
        Compute global average rating
        
        Args:
            sparse_matrix: Sparse rating matrix
            
        Returns:
            float: Global average rating
        """
        return sparse_matrix.sum() / sparse_matrix.count_nonzero()
    
    @staticmethod
    def create_feature_set(sparse_matrix):
        """
        Create complete feature set including averages
        
        Args:
            sparse_matrix: Training sparse matrix
            
        Returns:
            dict: Dictionary with global, user, and movie averages
        """
        averages = {}
        averages['global'] = FeatureEngineer.compute_global_average(sparse_matrix)
        averages['user'] = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=True)
        averages['movie'] = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=False)
        
        print(f"Global Average: {averages['global']:.3f}")
        print(f"User Averages computed: {len(averages['user'])}")
        print(f"Movie Averages computed: {len(averages['movie'])}")
        
        return averages


# ==================== SIMILARITY COMPUTATION ====================
class SimilarityComputer:
    """Computes similarity matrices for users and movies"""
    
    @staticmethod
    def compute_movie_similarity(sparse_matrix, filename='movie_movie_sim_sparse.npz'):
        """
        Compute movie-movie similarity matrix
        
        Args:
            sparse_matrix: User-Movie sparse matrix
            filename: File to save similarity matrix
            
        Returns:
            csr_matrix: Movie similarity matrix
        """
        if os.path.isfile(filename):
            print(f"Loading movie similarity from {filename}...")
            return sparse.load_npz(filename)
        
        print("Computing movie-movie similarity...")
        start = datetime.now()
        
        m_m_sim_sparse = cosine_similarity(X=sparse_matrix.T, dense_output=False)
        
        print(f"Done. Shape: {m_m_sim_sparse.shape}")
        print(f"Time taken: {datetime.now() - start}")
        
        sparse.save_npz(filename, m_m_sim_sparse)
        return m_m_sim_sparse
    
    @staticmethod
    def get_top_similar_items(similarity_matrix, top_n=100):
        """
        Get top N similar items for each item
        
        Args:
            similarity_matrix: Similarity matrix
            top_n: Number of top similar items
            
        Returns:
            dict: Dictionary of top similar items
        """
        print(f"Finding top {top_n} similar items...")
        start = datetime.now()
        
        item_ids = np.unique(similarity_matrix.nonzero()[1])
        similar_items = {}
        
        for item in item_ids:
            sim_items = similarity_matrix[item].toarray().ravel().argsort()[::-1][1:]
            similar_items[item] = sim_items[:top_n]
        
        print(f"Done. Time taken: {datetime.now() - start}")
        return similar_items


# ==================== MODEL TRAINING ====================
class ModelTrainer:
    """Trains and evaluates different recommendation models"""
    
    @staticmethod
    def get_error_metrics(y_true, y_pred):
        """
        Calculate RMSE and MAPE
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            tuple: (rmse, mape)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return rmse, mape
    
    @staticmethod
    def train_xgboost(x_train, y_train, x_test, y_test, tune_params=True):
        """
        Train XGBoost model with optional hyperparameter tuning
        
        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: Test labels
            tune_params: Whether to tune hyperparameters
            
        Returns:
            tuple: (model, train_results, test_results)
        """
        print("Training XGBoost model...")
        
        if tune_params:
            params = {
                'learning_rate': stats_uniform(0.01, 0.2),
                'n_estimators': sp_randint(100, 1000),
                'max_depth': sp_randint(1, 10),
                'min_child_weight': sp_randint(1, 8),
                'gamma': stats_uniform(0, 0.02),
                'subsample': stats_uniform(0.6, 0.4),
                'reg_alpha': sp_randint(0, 200),
                'reg_lambda': stats_uniform(0, 200),
                'colsample_bytree': stats_uniform(0.6, 0.3)
            }
            
            xgbreg = xgb.XGBRegressor(silent=True, n_jobs=-1, random_state=Config.RANDOM_SEED)
            
            print('Tuning hyperparameters...')
            xgb_search = RandomizedSearchCV(
                xgbreg, param_distributions=params,
                refit=False, scoring="neg_mean_squared_error",
                cv=3, n_jobs=-1, n_iter=10
            )
            xgb_search.fit(x_train, y_train)
            model = xgbreg.set_params(**xgb_search.best_params_)
        else:
            model = xgb.XGBRegressor(silent=True, n_jobs=-1, random_state=Config.RANDOM_SEED)
        
        # Train model
        start = datetime.now()
        model.fit(x_train, y_train, eval_metric='rmse')
        print(f'Training time: {datetime.now() - start}')
        
        # Evaluate
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        train_rmse, train_mape = ModelTrainer.get_error_metrics(y_train.values, y_train_pred)
        test_rmse, test_mape = ModelTrainer.get_error_metrics(y_test.values, y_test_pred)
        
        train_results = {'rmse': train_rmse, 'mape': train_mape, 'predictions': y_train_pred}
        test_results = {'rmse': test_rmse, 'mape': test_mape, 'predictions': y_test_pred}
        
        print(f'\nTest RMSE: {test_rmse:.4f}')
        print(f'Test MAPE: {test_mape:.2f}%')
        
        return model, train_results, test_results
    
    @staticmethod
    def train_surprise_model(algo, trainset, testset, verbose=True):
        """
        Train Surprise library model
        
        Args:
            algo: Surprise algorithm instance
            trainset: Training dataset
            testset: Test dataset
            verbose: Print progress
            
        Returns:
            tuple: (train_results, test_results)
        """
        print(f"Training {algo.__class__.__name__}...")
        start = datetime.now()
        
        # Train
        algo.fit(trainset)
        print(f'Training time: {datetime.now() - start}')
        
        # Get predictions
        train_preds = algo.test(trainset.build_testset())
        test_preds = algo.test(testset)
        
        # Extract ratings
        train_actual = np.array([pred.r_ui for pred in train_preds])
        train_pred = np.array([pred.est for pred in train_preds])
        test_actual = np.array([pred.r_ui for pred in test_preds])
        test_pred = np.array([pred.est for pred in test_preds])
        
        # Calculate errors
        train_rmse, train_mape = ModelTrainer.get_error_metrics(train_actual, train_pred)
        test_rmse, test_mape = ModelTrainer.get_error_metrics(test_actual, test_pred)
        
        if verbose:
            print(f'\nTrain RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%')
            print(f'Test RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%')
        
        train_results = {'rmse': train_rmse, 'mape': train_mape, 'predictions': train_pred}
        test_results = {'rmse': test_rmse, 'mape': test_mape, 'predictions': test_pred}
        
        return train_results, test_results


# ==================== VISUALIZATION ====================
class Visualizer:
    """Handles data visualization"""
    
    @staticmethod
    def plot_rating_distribution(df, title='Rating Distribution'):
        """Plot distribution of ratings"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='rating', data=df, ax=ax)
        plt.title(title, fontsize=15)
        plt.ylabel('Count')
        plt.xlabel('Rating')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results_dict, metric='rmse'):
        """
        Plot comparison of models
        
        Args:
            results_dict: Dictionary of model results
            metric: Metric to plot ('rmse' or 'mape')
        """
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        plt.figure(figsize=(12, 6))
        plt.bar(models, values)
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_train_test_comparison(train_results, test_results):
        """Plot train vs test performance"""
        models = list(train_results.keys())
        train_rmse = [train_results[m]['rmse'] for m in models]
        test_rmse = [test_results[m]['rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, train_rmse, width, label='Train')
        ax.bar(x + width/2, test_rmse, width, label='Test')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('RMSE')
        ax.set_title('Train vs Test RMSE')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# ==================== MAIN PIPELINE ====================
class NetflixRecommender:
    """Main recommendation system pipeline"""
    
    def __init__(self):
        self.train_sparse_matrix = None
        self.test_sparse_matrix = None
        self.train_averages = None
        self.models_evaluation_train = {}
        self.models_evaluation_test = {}
        
        # Set random seeds
        random.seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
    
    def run_full_pipeline(self, use_sample=True):
        """
        Run the complete recommendation pipeline
        
        Args:
            use_sample: Whether to use sampled data (recommended for testing)
        """
        print("="*60)
        print("Netflix Movie Recommendation System")
        print("="*60)
        
        # Step 1: Data Preprocessing
        print("\n[1/6] Data Preprocessing...")
        DataPreprocessor.merge_netflix_data()
        df = DataPreprocessor.load_and_sort_data()
        DataPreprocessor.check_data_quality(df)
        train_df, test_df = DataPreprocessor.split_train_test(df)
        
        # Step 2: Create Sparse Matrices
        print("\n[2/6] Creating Sparse Matrices...")
        self.train_sparse_matrix = SparseMatrixHandler.create_sparse_matrix(
            train_df, 'train_sparse_matrix.npz'
        )
        self.test_sparse_matrix = SparseMatrixHandler.create_sparse_matrix(
            test_df, 'test_sparse_matrix.npz'
        )
        
        SparseMatrixHandler.calculate_sparsity(self.train_sparse_matrix)
        
        # Step 3: Sample Data (if requested)
        if use_sample:
            print("\n[3/6] Sampling Data...")
            sample_train = SparseMatrixHandler.sample_sparse_matrix(
                self.train_sparse_matrix, 
                Config.SAMPLE_USERS, 
                Config.SAMPLE_MOVIES,
                'sample_train_sparse_matrix.npz'
            )
            sample_test = SparseMatrixHandler.sample_sparse_matrix(
                self.test_sparse_matrix,
                Config.SAMPLE_TEST_USERS,
                Config.SAMPLE_TEST_MOVIES,
                'sample_test_sparse_matrix.npz'
            )
            self.train_sparse_matrix = sample_train
            self.test_sparse_matrix = sample_test
        
        # Step 4: Feature Engineering
        print("\n[4/6] Feature Engineering...")
        self.train_averages = FeatureEngineer.create_feature_set(self.train_sparse_matrix)
        
        # Step 5: Compute Similarities
        print("\n[5/6] Computing Similarities...")
        movie_sim_matrix = SimilarityComputer.compute_movie_similarity(
            self.train_sparse_matrix
        )
        similar_movies = SimilarityComputer.get_top_similar_items(movie_sim_matrix)
        
        # Step 6: Model Training
        print("\n[6/6] Model Training Complete!")
        print("\nUse the train_models() method to train specific models.")
        print("Available models: baseline, knn, svd, svdpp, xgboost")
    
    def save_results(self, filename='model_results.csv'):
        """Save model evaluation results to CSV"""
        results_df = pd.DataFrame(self.models_evaluation_test)
        results_df.to_csv(filename)
        print(f"Results saved to {filename}")


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("Netflix Movie Recommendation Engine")
    print("====================================\n")
    
    # Initialize recommender
    recommender = NetflixRecommender()
    
    # Run pipeline
    recommender.run_full_pipeline(use_sample=True)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
