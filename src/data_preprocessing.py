"""
Data preprocessing module for Netflix Movie Recommendation Engine
Handles data loading, merging, cleaning, and train-test splitting
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from .config import Config


class DataPreprocessor:
    """Handles data loading, preprocessing, and transformation"""
    
    @staticmethod
    def merge_netflix_data(output_file=None, data_folder=None):
        """
        Merge all Netflix rating files into a single CSV file
        
        WHAT IT DOES (Simple Explanation):
        - Netflix data comes in 4 separate text files
        - This function reads all 4 files one by one
        - Combines them into a single, easy-to-use CSV file
        - Think of it like: merging 4 books into 1 master book
        
        HOW IT WORKS:
        1. Opens each file (combined_data_1.txt to combined_data_4.txt)
        2. Reads line by line
        3. When it sees "MovieID:", it remembers that movie
        4. For each rating line, adds the movie ID to it
        5. Writes everything to one big CSV file
        
        Args:
            output_file: Output CSV filename (default from Config)
            data_folder: Folder containing Netflix data files (default from Config)
            
        Returns:
            None (saves data to disk)
        """
        output_file = output_file or Config.MERGED_DATA_FILE
        data_folder = data_folder or Config.DATA_FOLDER
        
        if os.path.isfile(output_file):
            print(f"{output_file} already exists. Skipping merge.")
            return
        
        print("Merging Netflix data files...")
        start = datetime.now()
        
        with open(output_file, mode='w') as data:
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
                print("Done.")
        
        print(f'\nTotal time taken: {datetime.now() - start}')
    
    @staticmethod
    def load_and_sort_data(filename=None):
        """
        Load data from CSV and sort by date
        
        WHAT IT DOES (Simple Explanation):
        - Opens the merged CSV file we created
        - Loads all ratings into memory (like opening an Excel file)
        - Sorts all ratings by date (oldest first, newest last)
        - This is important because we want to predict FUTURE ratings
        
        WHY SORT BY DATE?
        - In real life, we know past ratings and predict future ones
        - Sorting ensures our train data is "old" and test data is "new"
        
        Args:
            filename: CSV file to load (default from Config)
            
        Returns:
            pd.DataFrame: Sorted dataframe (like a table with rows and columns)
        """
        filename = filename or Config.MERGED_DATA_FILE
        
        print(f"Loading dataframe from {filename}...")
        df = pd.read_csv(filename, sep=',', parse_dates=['date'], low_memory=False)
        
        print(f"Sorting by date...")
        df = df.sort_values(by='date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} ratings")
        return df
    
    @staticmethod
    def check_data_quality(df):
        """
        Check for NaN values and duplicates
        
        WHAT IT DOES (Simple Explanation):
        - Inspects the data for problems (like a quality inspector)
        - Checks if any data is missing (NaN = "Not a Number" = empty cells)
        - Looks for duplicate ratings (same user rating same movie twice)
        - Counts how many users, movies, and ratings we have
        
        WHY THIS MATTERS:
        - Missing data can crash our program
        - Duplicates can bias our model (giving too much weight to some ratings)
        - Knowing data size helps us plan (memory, time needed)
        
        EXAMPLE OUTPUT:
        - "We have 100 million ratings from 480K users on 17K movies"
        - "Found 50 duplicates and 0 missing values"
        
        Args:
            df: DataFrame to check
            
        Returns:
            dict: Quality metrics (numbers about data quality)
        """
        nan_count = sum(df.isnull().any())
        dup_count = sum(df.duplicated(['movie', 'user', 'rating']))
        
        metrics = {
            'nan_count': nan_count,
            'duplicates': dup_count,
            'n_users': df['user'].nunique(),
            'n_movies': df['movie'].nunique(),
            'n_ratings': len(df)
        }
        
        print(f"\nData Quality Check:")
        print(f"Number of users: {metrics['n_users']}")
        print(f"Number of movies: {metrics['n_movies']}")
        print(f"Number of ratings: {metrics['n_ratings']}")
        print(f"NaN values: {metrics['nan_count']}")
        print(f"Duplicate ratings: {metrics['duplicates']}\n")
        
        return metrics
    
    @staticmethod
    def split_train_test(df, train_ratio=None):
        """
        Split data into train and test sets based on time
        
        WHAT IT DOES (Simple Explanation):
        - Divides data into two parts: TRAIN (80%) and TEST (20%)
        - TRAIN = ratings we use to teach the computer
        - TEST = ratings we hide, then check if computer can predict them
        
        ANALOGY:
        - Like studying for an exam using practice problems (train)
        - Then taking the actual exam (test) to see how well you learned
        
        HOW IT SPLITS:
        - First 80% of ratings (by date) → Train set
        - Last 20% of ratings (by date) → Test set
        - This mimics real life: learn from past, predict future
        
        WHY 80-20?
        - 80% gives enough data to learn patterns
        - 20% gives enough data to reliably test performance
        - Industry standard split ratio
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for train split (default 0.80 = 80%)
            
        Returns:
            tuple: (train_df, test_df) - two separate dataframes
        """
        train_ratio = train_ratio or Config.TRAIN_SPLIT
        split_idx = int(df.shape[0] * train_ratio)
        
        train_file = Config.TRAIN_FILE
        test_file = Config.TEST_FILE
        
        if not os.path.isfile(train_file):
            df.iloc[:split_idx].to_csv(train_file, index=False)
            print(f"Train data saved to {train_file}")
        
        if not os.path.isfile(test_file):
            df.iloc[split_idx:].to_csv(test_file, index=False)
            print(f"Test data saved to {test_file}")
        
        train_df = pd.read_csv(train_file, parse_dates=['date'])
        test_df = pd.read_csv(test_file, parse_dates=['date'])
        
        print(f"\nTrain set: {len(train_df)} ratings")
        print(f"Test set: {len(test_df)} ratings\n")
        
        return train_df, test_df
    
    @staticmethod
    def load_movie_titles(filename=None):
        """
        Load movie titles and metadata
        
        Args:
            filename: Path to movie titles CSV
            
        Returns:
            pd.DataFrame: Movie metadata
        """
        filename = filename or Config.MOVIE_TITLES_FILE
        
        movie_titles = pd.read_csv(
            filename, sep=',', header=None,
            names=['movie_id', 'year_of_release', 'title'],
            index_col='movie_id', encoding="ISO-8859-1"
        )
        
        print(f"Loaded {len(movie_titles)} movies")
        return movie_titles
