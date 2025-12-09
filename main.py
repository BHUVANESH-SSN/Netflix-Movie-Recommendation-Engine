"""
Main execution script for Netflix Movie Recommendation Engine
Demonstrates the complete pipeline for movie recommendations
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.sparse_matrix_handler import SparseMatrixHandler
from src.feature_engineering import FeatureEngineer
from src.similarity import SimilarityComputer
from src.models import ModelTrainer
from src.visualization import Visualizer


def run_data_pipeline():
    """Execute data preprocessing pipeline"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    # Merge data files
    DataPreprocessor.merge_netflix_data()
    
    # Load and sort data
    df = DataPreprocessor.load_and_sort_data()
    
    # Check data quality
    DataPreprocessor.check_data_quality(df)
    
    # Split into train and test
    train_df, test_df = DataPreprocessor.split_train_test(df)
    
    return df, train_df, test_df


def create_sparse_matrices(train_df, test_df, use_sample=True):
    """Create sparse matrices from dataframes"""
    print("\n" + "="*70)
    print("STEP 2: CREATING SPARSE MATRICES")
    print("="*70)
    
    # Create training sparse matrix
    train_sparse = SparseMatrixHandler.create_sparse_matrix(
        train_df, Config.TRAIN_SPARSE_MATRIX
    )
    SparseMatrixHandler.print_matrix_info(train_sparse, "Training Matrix")
    
    # Create test sparse matrix
    test_sparse = SparseMatrixHandler.create_sparse_matrix(
        test_df, Config.TEST_SPARSE_MATRIX
    )
    SparseMatrixHandler.print_matrix_info(test_sparse, "Test Matrix")
    
    # Sample if requested
    if use_sample:
        print("\nSampling data for faster training...")
        train_sparse = SparseMatrixHandler.sample_sparse_matrix(
            train_sparse,
            Config.SAMPLE_USERS,
            Config.SAMPLE_MOVIES,
            'sample_train_sparse_matrix.npz'
        )
        
        test_sparse = SparseMatrixHandler.sample_sparse_matrix(
            test_sparse,
            Config.SAMPLE_TEST_USERS,
            Config.SAMPLE_TEST_MOVIES,
            'sample_test_sparse_matrix.npz'
        )
    
    return train_sparse, test_sparse


def engineer_features(train_sparse):
    """Create feature set"""
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    train_averages = FeatureEngineer.create_feature_set(train_sparse)
    return train_averages


def compute_similarities(train_sparse):
    """Compute similarity matrices"""
    print("\n" + "="*70)
    print("STEP 4: COMPUTING SIMILARITIES")
    print("="*70)
    
    # Movie-movie similarity
    movie_sim = SimilarityComputer.compute_movie_similarity(train_sparse)
    similar_movies = SimilarityComputer.get_top_similar_items(movie_sim)
    
    print(f"Similarity matrix shape: {movie_sim.shape}")
    print(f"Top similar movies computed for {len(similar_movies)} movies")
    
    return movie_sim, similar_movies


def demonstrate_similarity(movie_sim):
    """Demonstrate similarity computation"""
    print("\n" + "="*70)
    print("STEP 5: SIMILARITY DEMONSTRATION")
    print("="*70)
    
    # Load movie titles
    try:
        movie_titles = DataPreprocessor.load_movie_titles()
        
        # Show similar movies for a random movie
        movie_id = 67  # Vampire Journals
        SimilarityComputer.print_similar_movies(
            movie_id, movie_sim, movie_titles, top_n=10
        )
    except Exception as e:
        print(f"Could not demonstrate similarity: {e}")


def train_models():
    """Train recommendation models"""
    print("\n" + "="*70)
    print("STEP 6: MODEL TRAINING")
    print("="*70)
    
    print("\nNote: Model training requires prepared feature sets.")
    print("Please refer to the notebook for complete model training pipeline.")
    print("\nAvailable models:")
    print("  - Baseline Model")
    print("  - KNN Baseline (User-User)")
    print("  - KNN Baseline (Item-Item)")
    print("  - SVD Matrix Factorization")
    print("  - SVD++ with Implicit Feedback")
    print("  - XGBoost Ensemble")


def main(use_sample=True):
    """
    Main execution function
    
    Args:
        use_sample: Use sampled data for faster execution
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("NETFLIX MOVIE RECOMMENDATION ENGINE")
    print("="*70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using Sample Data: {use_sample}")
    print("="*70)
    
    try:
        # Step 1: Data preprocessing
        df, train_df, test_df = run_data_pipeline()
        
        # Step 2: Create sparse matrices
        train_sparse, test_sparse = create_sparse_matrices(
            train_df, test_df, use_sample
        )
        
        # Step 3: Feature engineering
        train_averages = engineer_features(train_sparse)
        
        # Step 4: Compute similarities
        movie_sim, similar_movies = compute_similarities(train_sparse)
        
        # Step 5: Demonstrate similarity
        demonstrate_similarity(movie_sim)
        
        # Step 6: Model training info
        train_models()
        
        # Success message
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")
        print("="*70)
        
        print("\nNext Steps:")
        print("  1. Review the generated sparse matrices")
        print("  2. Examine feature sets and averages")
        print("  3. Use the notebook for complete model training")
        print("  4. Evaluate and compare different models")
        print("\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}\n")
        raise


if __name__ == "__main__":
    # Run with sample data by default (faster for testing)
    # Set to False to use full dataset (requires significant time and memory)
    main(use_sample=True)
