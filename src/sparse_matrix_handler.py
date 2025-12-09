"""
Sparse matrix operations module
Handles creation, loading, saving, and sampling of sparse matrices
"""

import os
from datetime import datetime
import numpy as np
from scipy import sparse
from .config import Config


class SparseMatrixHandler:
    """Handles sparse matrix creation and operations"""
    
    @staticmethod
    def create_sparse_matrix(df, filename=None):
        """
        Create sparse matrix from dataframe
        
        WHAT IT DOES (Simple Explanation):
        - Converts our ratings data into a special grid/table format
        - ROWS = Users, COLUMNS = Movies, CELLS = Ratings
        - Empty cells = movies user hasn't rated (most cells are empty!)
        
        EXAMPLE VISUALIZATION:
                Movie1  Movie2  Movie3  Movie4
        User1     5       0       0       3
        User2     0       4       0       0
        User3     0       0       5       4
        
        WHY "SPARSE"?
        - 480K users × 17K movies = 8 BILLION possible cells
        - But only 100M ratings = 98.8% cells are empty (zeros)
        - Sparse matrix only stores non-zero values (saves HUGE memory)
        - Instead of 60GB, we use only 2GB!
        
        HOW IT SAVES MEMORY:
        - Normal matrix: stores ALL cells (including millions of zeros)
        - Sparse matrix: only stores (user, movie, rating) for non-zeros
        
        Args:
            df: DataFrame with user, movie, rating columns
            filename: Optional filename to save/load matrix
            
        Returns:
            csr_matrix: Sparse matrix (users x movies) - compressed format
        """
        if filename and os.path.isfile(filename):
            print(f"Loading sparse matrix from {filename}...")
            return sparse.load_npz(filename)
        
        print("Creating sparse matrix from dataframe...")
        start = datetime.now()
        
        sparse_matrix = sparse.csr_matrix(
            (df.rating.values, (df.user.values, df.movie.values))
        )
        
        print(f'Shape: {sparse_matrix.shape} (users x movies)')
        print(f'Non-zero elements: {sparse_matrix.count_nonzero():,}')
        print(f'Time taken: {datetime.now() - start}')
        
        if filename:
            print(f'Saving to {filename}...')
            sparse.save_npz(filename, sparse_matrix)
            print('Done.\n')
        
        return sparse_matrix
    
    @staticmethod
    def calculate_sparsity(sparse_matrix):
        """
        Calculate sparsity percentage of matrix
        
        WHAT IT DOES (Simple Explanation):
        - Calculates what percentage of the matrix is empty (zeros)
        - Tells us how "sparse" (empty) our data is
        
        THE MATH (Simple):
        - Count total possible cells = users × movies
        - Count actual ratings (non-zero cells)
        - Sparsity = (empty cells / total cells) × 100
        
        EXAMPLE:
        - Total cells: 1000
        - Filled cells: 20
        - Empty cells: 980
        - Sparsity = 980/1000 × 100 = 98%
        
        WHY IT MATTERS:
        - High sparsity (98%+) means sparse matrix is ESSENTIAL
        - Shows why normal matrix would waste memory
        - Helps us choose right algorithms (some work better with sparse data)
        
        Args:
            sparse_matrix: Sparse matrix
            
        Returns:
            float: Sparsity percentage (e.g., 98.8%)
        """
        users, movies = sparse_matrix.shape
        elements = sparse_matrix.count_nonzero()
        sparsity = (1 - (elements / (users * movies))) * 100
        
        print(f"Matrix Sparsity: {sparsity:.4f}%")
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
        
        print(f"\nCreating sample with {no_users} users and {no_movies} movies...")
        
        row_ind, col_ind, ratings = sparse.find(sparse_matrix)
        users = np.unique(row_ind)
        movies = np.unique(col_ind)
        
        print(f"Original: {len(users)} users, {len(movies)} movies, {len(ratings)} ratings")
        
        # Sample with fixed seed for reproducibility
        np.random.seed(Config.RANDOM_SEED)
        sample_users = np.random.choice(users, no_users, replace=False)
        sample_movies = np.random.choice(movies, no_movies, replace=False)
        
        # Create mask for sampled data
        mask = np.logical_and(
            np.isin(row_ind, sample_users),
            np.isin(col_ind, sample_movies)
        )
        
        # Create sampled matrix
        sample_sparse_matrix = sparse.csr_matrix(
            (ratings[mask], (row_ind[mask], col_ind[mask])),
            shape=(max(sample_users) + 1, max(sample_movies) + 1)
        )
        
        print(f'Sampled: {ratings[mask].shape[0]} ratings')
        
        # Save to disk
        sparse.save_npz(path, sample_sparse_matrix)
        print(f'Saved to {path}\n')
        
        return sample_sparse_matrix
    
    @staticmethod
    def get_matrix_stats(sparse_matrix):
        """
        Get comprehensive statistics about sparse matrix
        
        Args:
            sparse_matrix: Sparse matrix
            
        Returns:
            dict: Statistics dictionary
        """
        users, movies = sparse_matrix.shape
        elements = sparse_matrix.count_nonzero()
        sparsity = (1 - (elements / (users * movies))) * 100
        
        stats = {
            'shape': sparse_matrix.shape,
            'users': users,
            'movies': movies,
            'ratings': elements,
            'sparsity': sparsity,
            'density': 100 - sparsity,
            'avg_ratings_per_user': elements / users if users > 0 else 0,
            'avg_ratings_per_movie': elements / movies if movies > 0 else 0
        }
        
        return stats
    
    @staticmethod
    def print_matrix_info(sparse_matrix, name="Matrix"):
        """
        Print detailed information about sparse matrix
        
        Args:
            sparse_matrix: Sparse matrix
            name: Name of the matrix for display
        """
        stats = SparseMatrixHandler.get_matrix_stats(sparse_matrix)
        
        print(f"\n{name} Information:")
        print("-" * 50)
        print(f"Shape (users x movies): {stats['shape']}")
        print(f"Total Users: {stats['users']:,}")
        print(f"Total Movies: {stats['movies']:,}")
        print(f"Total Ratings: {stats['ratings']:,}")
        print(f"Sparsity: {stats['sparsity']:.4f}%")
        print(f"Density: {stats['density']:.4f}%")
        print(f"Avg Ratings per User: {stats['avg_ratings_per_user']:.2f}")
        print(f"Avg Ratings per Movie: {stats['avg_ratings_per_movie']:.2f}")
        print("-" * 50 + "\n")
