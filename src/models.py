"""
Model training and evaluation module
Contains model trainers for XGBoost and Surprise library models
"""

import numpy as np
from datetime import datetime
from scipy.stats import randint as sp_randint, uniform as stats_uniform
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from surprise import BaselineOnly, KNNBaseline, SVD, SVDpp
from .config import Config


class ModelTrainer:
    """Trains and evaluates different recommendation models"""
    
    @staticmethod
    def get_error_metrics(y_true, y_pred):
        """
        Calculate RMSE and MAPE
        
        WHAT IT DOES (Simple Explanation):
        - Measures how good our predictions are
        - Compares what we predicted vs. what actually happened
        - Gives us TWO error scores: RMSE and MAPE
        
        RMSE (Root Mean Square Error):
        - Average error in rating points
        - Example: RMSE = 0.95 means we're off by ~0.95 stars on average
        - Lower is better (0 = perfect predictions)
        
        RMSE EXAMPLE:
        - Actual: 4 stars, Predicted: 3 stars → Error = 1.0
        - Actual: 5 stars, Predicted: 4.5 stars → Error = 0.5
        - RMSE = sqrt(average of squared errors)
        
        MAPE (Mean Absolute Percentage Error):
        - Error as a percentage
        - Example: MAPE = 18% means we're off by 18% on average
        - Lower is better (0% = perfect predictions)
        
        MAPE EXAMPLE:
        - Actual: 4 stars, Predicted: 3 stars → Error = 25%
        - Actual: 5 stars, Predicted: 4.5 stars → Error = 10%
        - MAPE = average of percentage errors
        
        WHY TWO METRICS?
        - RMSE: good for comparing models (standard metric)
        - MAPE: easy to understand ("We're 18% off")
        - Both tell slightly different stories about errors
        
        Args:
            y_true: True ratings (what users actually gave)
            y_pred: Predicted ratings (what our model predicted)
            
        Returns:
            tuple: (rmse, mape) - two numbers measuring prediction quality
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return rmse, mape
    
    @staticmethod
    def train_xgboost(x_train, y_train, x_test, y_test, 
                     tune_params=True, n_iter=10, verbose=True):
        """
        Train XGBoost model with optional hyperparameter tuning
        
        WHAT IT DOES (Simple Explanation):
        - Trains a powerful prediction model called XGBoost
        - XGBoost = "Extreme Gradient Boosting" = ensemble of decision trees
        - Think of it as: combining many "weak" predictors into one "strong" predictor
        
        HOW XGBOOST WORKS (Analogy):
        - Like asking 100 people to guess a number
        - First person guesses, makes some errors
        - Second person tries to fix first person's errors
        - Third person tries to fix remaining errors
        - Continue 100 times
        - Final answer = combination of all 100 guesses
        
        WHAT IS "HYPERPARAMETER TUNING"?
        - Finding best settings for the model (like tuning a radio)
        - Example settings:
          * How many trees? (more trees = more complex)
          * How deep each tree? (deeper = more detailed patterns)
          * Learning rate? (how fast to learn)
        - Tries different combinations, picks the best one
        
        THE PROCESS:
        1. Try different parameter combinations (random search)
        2. For each combination, train model and check accuracy
        3. Pick combination with best accuracy
        4. Train final model with best parameters
        5. Test on unseen data
        
        WHY XGBOOST IS POWERFUL:
        - Handles complex patterns automatically
        - Combines many features intelligently
        - Often wins machine learning competitions
        - Great at not overfitting (memorizing training data)
        
        Args:
            x_train: Training features (our input data)
            y_train: Training labels (correct answers)
            x_test: Test features (new data to predict)
            y_test: Test labels (answers we check against)
            tune_params: Whether to tune hyperparameters (recommended!)
            n_iter: Number of tuning iterations (more = better but slower)
            verbose: Print detailed output
            
        Returns:
            tuple: (model, train_results, test_results)
                   - model: trained XGBoost model
                   - train_results: how well it did on training data
                   - test_results: how well it did on test data (most important!)
        """
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60 + "\n")
        
        model = xgb.XGBRegressor(
            silent=True, 
            n_jobs=-1, 
            random_state=Config.RANDOM_SEED
        )
        
        if tune_params:
            print(f'Hyperparameter tuning with {n_iter} iterations...')
            
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
            
            xgb_search = RandomizedSearchCV(
                model, param_distributions=params,
                refit=False, scoring="neg_mean_squared_error",
                cv=3, n_jobs=-1, n_iter=n_iter, verbose=0
            )
            
            start = datetime.now()
            xgb_search.fit(x_train, y_train)
            print(f'Tuning time: {datetime.now() - start}')
            print(f'Best RMSE: {np.sqrt(-xgb_search.best_score_):.4f}\n')
            
            model = model.set_params(**xgb_search.best_params_)
        
        # Train model
        print('Training final model...')
        start = datetime.now()
        model.fit(x_train, y_train, eval_metric='rmse', verbose=False)
        print(f'Training time: {datetime.now() - start}')
        
        # Evaluate on train set
        y_train_pred = model.predict(x_train)
        train_rmse, train_mape = ModelTrainer.get_error_metrics(
            y_train.values, y_train_pred
        )
        
        # Evaluate on test set
        y_test_pred = model.predict(x_test)
        test_rmse, test_mape = ModelTrainer.get_error_metrics(
            y_test.values, y_test_pred
        )
        
        if verbose:
            print(f'\nTrain RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%')
            print(f'Test RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%')
        
        train_results = {
            'rmse': train_rmse, 
            'mape': train_mape, 
            'predictions': y_train_pred
        }
        test_results = {
            'rmse': test_rmse, 
            'mape': test_mape, 
            'predictions': y_test_pred
        }
        
        return model, train_results, test_results
    
    @staticmethod
    def train_surprise_model(algo, trainset, testset, model_name="Model", verbose=True):
        """
        Train Surprise library model
        
        WHAT IT DOES (Simple Explanation):
        - Trains recommendation models using the "Surprise" library
        - Surprise = Python library specialized for recommendation systems
        - Can train different types of models: Baseline, KNN, SVD, SVD++
        
        WHAT IS THE SURPRISE LIBRARY?
        - Like a toolbox specifically for building recommendation systems
        - Has pre-built algorithms we can use
        - Saves us from coding complex math from scratch
        
        MODELS IT CAN TRAIN:
        
        1. BASELINE MODEL:
           - Simplest model
           - Prediction = Global Average + User Bias + Movie Bias
           - Example: "User likes action (+0.5), Movie is popular (+0.3)"
        
        2. KNN (K-Nearest Neighbors):
           - "Birds of a feather flock together"
           - Finds similar users/movies and uses their ratings
           - Example: "Users like you rated this movie 4.5 stars"
        
        3. SVD (Matrix Factorization):
           - Discovers hidden patterns (like genres, themes)
           - Maps users and movies to "concept space"
           - Example: "This is 80% action, 20% comedy movie"
        
        4. SVD++:
           - Enhanced SVD
           - Also considers WHICH movies user rated (not just ratings)
           - Example: "You rated many sci-fi movies, even if ratings vary"
        
        THE PROCESS:
        1. Load the algorithm (Baseline/KNN/SVD/SVD++)
        2. Feed it training data
        3. Model learns patterns
        4. Test on new data
        5. Calculate accuracy (RMSE, MAPE)
        
        Args:
            algo: Surprise algorithm instance (which model to use)
            trainset: Training dataset (Surprise format)
            testset: Test dataset (Surprise format)
            model_name: Name of the model for display
            verbose: Print detailed output
            
        Returns:
            tuple: (train_results, test_results)
                   - How well model performed on both datasets
        """
        print("\n" + "="*60)
        print(f"Training {model_name}")
        print("="*60 + "\n")
        
        # Train
        start = datetime.now()
        algo.fit(trainset)
        print(f'Training time: {datetime.now() - start}')
        
        # Get predictions
        print('Evaluating on train set...')
        train_preds = algo.test(trainset.build_testset())
        
        print('Evaluating on test set...')
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
        
        train_results = {
            'rmse': train_rmse, 
            'mape': train_mape, 
            'predictions': train_pred
        }
        test_results = {
            'rmse': test_rmse, 
            'mape': test_mape, 
            'predictions': test_pred
        }
        
        return train_results, test_results
    
    @staticmethod
    def train_baseline_model(trainset, testset):
        """Train Surprise Baseline model"""
        bsl_options = {'method': 'sgd', 'learning_rate': 0.001}
        algo = BaselineOnly(bsl_options=bsl_options)
        return ModelTrainer.train_surprise_model(
            algo, trainset, testset, "Baseline Model"
        )
    
    @staticmethod
    def train_knn_baseline(trainset, testset, user_based=True, k=None):
        """
        Train KNN Baseline model
        
        Args:
            trainset: Training dataset
            testset: Test dataset
            user_based: True for user-user, False for item-item
            k: Number of neighbors (default from Config)
        """
        k = k or Config.KNN_NEIGHBORS
        
        sim_options = {
            'user_based': user_based,
            'name': 'pearson_baseline',
            'shrinkage': 100,
            'min_support': 2
        }
        bsl_options = {'method': 'sgd'}
        
        algo = KNNBaseline(k=k, sim_options=sim_options, bsl_options=bsl_options)
        
        model_type = "User-User" if user_based else "Item-Item"
        return ModelTrainer.train_surprise_model(
            algo, trainset, testset, f"KNN Baseline ({model_type})"
        )
    
    @staticmethod
    def train_svd(trainset, testset, n_factors=None):
        """
        Train SVD model
        
        Args:
            trainset: Training dataset
            testset: Test dataset
            n_factors: Number of latent factors (default from Config)
        """
        n_factors = n_factors or Config.SVD_FACTORS
        
        algo = SVD(
            n_factors=n_factors, 
            biased=True, 
            random_state=Config.RANDOM_SEED,
            verbose=False
        )
        
        return ModelTrainer.train_surprise_model(
            algo, trainset, testset, f"SVD (factors={n_factors})"
        )
    
    @staticmethod
    def train_svdpp(trainset, testset, n_factors=None):
        """
        Train SVD++ model
        
        Args:
            trainset: Training dataset
            testset: Test dataset
            n_factors: Number of latent factors (default from Config)
        """
        n_factors = n_factors or Config.SVDPP_FACTORS
        
        algo = SVDpp(
            n_factors=n_factors,
            random_state=Config.RANDOM_SEED,
            verbose=False
        )
        
        return ModelTrainer.train_surprise_model(
            algo, trainset, testset, f"SVD++ (factors={n_factors})"
        )
