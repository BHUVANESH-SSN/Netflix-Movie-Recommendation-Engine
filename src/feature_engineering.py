"""
Feature engineering module
Computes various features like averages, similarities, and derived metrics
"""

import numpy as np
from datetime import datetime


class FeatureEngineer:
    """Handles feature engineering for ML models"""
    
    @staticmethod
    def get_average_ratings(sparse_matrix, of_users=True):
        """
        Calculate average ratings per user or per movie
        
        WHAT IT DOES (Simple Explanation):
        - Calculates the average (mean) rating for each user OR each movie
        - If of_users=True: "What's each user's typical rating?"
        - If of_users=False: "What's each movie's typical rating?"
        
        USER AVERAGE EXAMPLE:
        - User123 rated: [5, 4, 5, 3, 4]
        - Average = (5+4+5+3+4)/5 = 4.2
        - Meaning: User123 typically gives ~4-star ratings
        
        MOVIE AVERAGE EXAMPLE:
        - Movie456 received: [5, 3, 4, 5, 4]
        - Average = (5+3+4+5+4)/5 = 4.2
        - Meaning: Movie456 is typically rated ~4 stars (pretty good!)
        
        WHY WE NEED THIS:
        - Some users are "harsh critics" (average 2 stars)
        - Some users are "generous" (average 4.5 stars)
        - Some movies are "masterpieces" (average 4.8 stars)
        - Some movies are "flops" (average 2.1 stars)
        - Helps us make better predictions by knowing these patterns
        
        Args:
            sparse_matrix: Sparse rating matrix
            of_users: True for user averages, False for movie averages
            
        Returns:
            dict: Dictionary mapping user/movie ID to average rating
                  Example: {User1: 4.2, User2: 3.8, ...}
        """
        axis = 1 if of_users else 0
        entity_type = "users" if of_users else "movies"
        
        print(f"Computing average ratings for {entity_type}...")
        start = datetime.now()
        
        sum_of_ratings = sparse_matrix.sum(axis=axis).A1
        is_rated = sparse_matrix != 0
        no_of_ratings = is_rated.sum(axis=axis).A1
        
        u, m = sparse_matrix.shape
        average_ratings = {
            i: sum_of_ratings[i] / no_of_ratings[i]
            for i in range(u if of_users else m)
            if no_of_ratings[i] != 0
        }
        
        print(f"Computed averages for {len(average_ratings)} {entity_type}")
        print(f"Time taken: {datetime.now() - start}\n")
        
        return average_ratings
    
    @staticmethod
    def compute_global_average(sparse_matrix):
        """
        Compute global average rating
        
        WHAT IT DOES (Simple Explanation):
        - Calculates the average of ALL ratings in the entire dataset
        - ONE single number that represents "typical rating"
        
        THE MATH:
        - Add up ALL ratings: 1+5+3+4+2+...+4 = Sum
        - Divide by total number of ratings
        - Result: Global Average (usually around 3.5-3.7 for Netflix)
        
        EXAMPLE:
        - Total ratings: 100,000,000
        - Sum of all ratings: 360,000,000
        - Global average = 360M / 100M = 3.6 stars
        
        WHY WE NEED THIS:
        - Simplest possible prediction: "Every movie gets 3.6 stars"
        - Used as fallback when we have NO information about user or movie
        - Baseline for comparison (our models should beat this!)
        
        WHEN IT'S USED:
        - Brand new user who hasn't rated anything yet
        - Brand new movie with no ratings yet
        - "Cold start" problem solution
        
        Args:
            sparse_matrix: Sparse rating matrix
            
        Returns:
            float: Global average rating (e.g., 3.6)
        """
        global_avg = sparse_matrix.sum() / sparse_matrix.count_nonzero()
        print(f"Global Average Rating: {global_avg:.4f}")
        return global_avg
    
    @staticmethod
    def create_feature_set(sparse_matrix):
        """
        Create complete feature set including all averages
        
        Args:
            sparse_matrix: Training sparse matrix
            
        Returns:
            dict: Dictionary with global, user, and movie averages
        """
        print("\n" + "="*60)
        print("Creating Feature Set")
        print("="*60 + "\n")
        
        averages = {}
        
        # Global average
        averages['global'] = FeatureEngineer.compute_global_average(sparse_matrix)
        
        # User averages
        averages['user'] = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=True)
        
        # Movie averages
        averages['movie'] = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=False)
        
        print("Feature Set Summary:")
        print("-" * 50)
        print(f"Global Average: {averages['global']:.4f}")
        print(f"User Averages: {len(averages['user'])} users")
        print(f"Movie Averages: {len(averages['movie'])} movies")
        print("-" * 50 + "\n")
        
        return averages
    
    @staticmethod
    def analyze_cold_start(full_df, train_averages):
        """
        Analyze cold start problem - new users/movies not in training
        
        Args:
            full_df: Complete dataframe with all data
            train_averages: Training averages dictionary
            
        Returns:
            dict: Cold start statistics
        """
        total_users = len(np.unique(full_df.user))
        total_movies = len(np.unique(full_df.movie))
        
        train_users = len(train_averages['user'])
        train_movies = len(train_averages['movie'])
        
        new_users = total_users - train_users
        new_movies = total_movies - train_movies
        
        cold_start_stats = {
            'total_users': total_users,
            'train_users': train_users,
            'new_users': new_users,
            'new_users_pct': (new_users / total_users) * 100,
            'total_movies': total_movies,
            'train_movies': train_movies,
            'new_movies': new_movies,
            'new_movies_pct': (new_movies / total_movies) * 100
        }
        
        print("\nCold Start Analysis:")
        print("-" * 50)
        print(f"Total Users: {total_users:,}")
        print(f"Users in Training: {train_users:,}")
        print(f"New Users (Cold Start): {new_users:,} ({cold_start_stats['new_users_pct']:.2f}%)")
        print()
        print(f"Total Movies: {total_movies:,}")
        print(f"Movies in Training: {train_movies:,}")
        print(f"New Movies (Cold Start): {new_movies:,} ({cold_start_stats['new_movies_pct']:.2f}%)")
        print("-" * 50 + "\n")
        
        return cold_start_stats
    
    @staticmethod
    def get_rating_statistics(sparse_matrix):
        """
        Get statistical measures of ratings
        
        Args:
            sparse_matrix: Sparse rating matrix
            
        Returns:
            dict: Statistics dictionary
        """
        ratings = sparse_matrix.data
        
        stats = {
            'mean': np.mean(ratings),
            'median': np.median(ratings),
            'std': np.std(ratings),
            'min': np.min(ratings),
            'max': np.max(ratings),
            'q25': np.percentile(ratings, 25),
            'q75': np.percentile(ratings, 75)
        }
        
        return stats
