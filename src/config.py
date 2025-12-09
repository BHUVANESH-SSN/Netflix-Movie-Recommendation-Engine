"""
Configuration module for Netflix Movie Recommendation Engine
Contains all constants and configuration parameters
"""


class Config:
    """Configuration constants for the recommendation system"""
    
    # Data paths
    DATA_FOLDER = 'data_folder'
    COMBINED_FILES = ['combined_data_1.txt', 'combined_data_2.txt', 
                      'combined_data_3.txt', 'combined_data_4.txt']
    MOVIE_TITLES_FILE = 'data_folder/movie_titles.csv'
    
    # File names
    MERGED_DATA_FILE = 'data.csv'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    TRAIN_SPARSE_MATRIX = 'train_sparse_matrix.npz'
    TEST_SPARSE_MATRIX = 'test_sparse_matrix.npz'
    MOVIE_SIMILARITY_MATRIX = 'movie_movie_sim_sparse.npz'
    
    # Split ratios
    TRAIN_SPLIT = 0.80
    TEST_SPLIT = 0.20
    
    # Random seed for reproducibility
    RANDOM_SEED = 15
    
    # Sampling parameters
    SAMPLE_USERS = 10000
    SAMPLE_MOVIES = 1000
    SAMPLE_TEST_USERS = 5000
    SAMPLE_TEST_MOVIES = 500
    
    # Model parameters
    XGBOOST_PARAMS = {
        'learning_rate': (0.01, 0.2),
        'n_estimators': (100, 1000),
        'max_depth': (1, 10),
        'min_child_weight': (1, 8),
        'gamma': (0, 0.02),
        'subsample': (0.6, 0.4),
        'reg_alpha': (0, 200),
        'reg_lambda': (0, 200),
        'colsample_bytree': (0.6, 0.3)
    }
    
    # Similarity parameters
    TOP_N_SIMILAR = 100
    KNN_NEIGHBORS = 40
    SVD_FACTORS = 100
    SVDPP_FACTORS = 50
