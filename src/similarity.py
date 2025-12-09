"""
Similarity computation module
Computes user-user and movie-movie similarities
"""

import os
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from .config import Config


class SimilarityComputer:
    """Computes similarity matrices for users and movies"""
    
    @staticmethod
    def compute_movie_similarity(sparse_matrix, filename=None):
        """
        Compute movie-movie similarity matrix using cosine similarity
        
        WHAT IT DOES (Simple Explanation):
        - Finds which movies are similar to each other
        - Creates a "similarity score" between every pair of movies
        - Score ranges from 0 (totally different) to 1 (very similar)
        
        HOW IT DETERMINES SIMILARITY:
        - If same users like both movies → movies are similar
        - Example: Users who like "Avengers" also like "Iron Man"
        - Uses "Cosine Similarity" (measures angle between vectors)
        
        REAL EXAMPLE:
        - Movie: "The Matrix"
        - Similar movies found:
          * "Matrix Reloaded" (similarity: 0.95)
          * "Inception" (similarity: 0.78)
          * "Blade Runner" (similarity: 0.65)
        
        THE MATH (Simplified):
        - Compare rating patterns of two movies
        - If rating patterns match → high similarity
        - If patterns differ → low similarity
        
        WHY IT'S USEFUL:
        - "If you liked Matrix, you'll probably like Inception"
        - Item-based collaborative filtering
        - Powers "Similar Movies" recommendations
        
        Args:
            sparse_matrix: User-Movie sparse matrix
            filename: File to save similarity matrix
            
        Returns:
            csr_matrix: Movie similarity matrix (movies × movies)
        """
        filename = filename or Config.MOVIE_SIMILARITY_MATRIX
        
        if os.path.isfile(filename):
            print(f"Loading movie similarity from {filename}...")
            return sparse.load_npz(filename)
        
        print("Computing movie-movie similarity matrix...")
        start = datetime.now()
        
        # Transpose to get movies x users, then compute cosine similarity
        m_m_sim_sparse = cosine_similarity(X=sparse_matrix.T, dense_output=False)
        
        print(f"Similarity matrix shape: {m_m_sim_sparse.shape}")
        print(f"Time taken: {datetime.now() - start}")
        
        # Save to disk
        print(f"Saving to {filename}...")
        sparse.save_npz(filename, m_m_sim_sparse)
        print("Done.\n")
        
        return m_m_sim_sparse
    
    @staticmethod
    def compute_user_similarity(sparse_matrix, num_users=100, top_k=100):
        """
        Compute user-user similarity for a subset of users
        Note: Computing for all users is computationally expensive
        
        Args:
            sparse_matrix: User-Movie sparse matrix
            num_users: Number of users to compute similarity for
            top_k: Number of top similar users to keep
            
        Returns:
            csr_matrix: User similarity matrix (sparse)
        """
        print(f"Computing user-user similarity for {num_users} users...")
        print(f"Keeping top {top_k} similar users for each user")
        start = datetime.now()
        
        no_of_users, _ = sparse_matrix.shape
        row_ind, col_ind = sparse_matrix.nonzero()
        row_ind = sorted(set(row_ind))
        
        rows, cols, data = [], [], []
        
        for i, row in enumerate(row_ind[:num_users]):
            # Compute similarity for this user with all others
            sim = cosine_similarity(sparse_matrix.getrow(row), sparse_matrix).ravel()
            
            # Get top k similar users
            top_sim_ind = sim.argsort()[-top_k:]
            top_sim_val = sim[top_sim_ind]
            
            rows.extend([row] * top_k)
            cols.extend(top_sim_ind)
            data.extend(top_sim_val)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{num_users} users...")
        
        # Create sparse similarity matrix
        u_u_sim_sparse = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(no_of_users, no_of_users)
        )
        
        print(f"Done. Time taken: {datetime.now() - start}\n")
        return u_u_sim_sparse
    
    @staticmethod
    def get_top_similar_items(similarity_matrix, top_n=None):
        """
        Get top N similar items for each item
        
        WHAT IT DOES (Simple Explanation):
        - For each movie, finds its "best friends" (most similar movies)
        - Keeps only the TOP most similar ones (default: top 100)
        - Throws away less similar ones to save memory
        
        ANALOGY:
        - You have 1000 acquaintances
        - But you only keep in touch with your top 10 closest friends
        - Same idea: we don't need ALL similarities, just the strongest ones
        
        WHY ONLY TOP N?
        - Storing ALL similarities = too much memory
        - Low similarities aren't useful (who cares if two movies are 1% similar?)
        - Top 100 similar movies is more than enough for recommendations
        
        EXAMPLE OUTPUT:
        For "The Matrix":
        - Similar movie 1: "Matrix Reloaded" (rank #1)
        - Similar movie 2: "Inception" (rank #2)
        - ...
        - Similar movie 100: "Blade Runner" (rank #100)
        
        REAL USE CASE:
        - User just watched "The Matrix"
        - Show them top 10 similar movies
        - Don't need to check all 17,000 movies, just the top 100
        
        Args:
            similarity_matrix: Similarity matrix (items x items)
            top_n: Number of top similar items (default 100)
            
        Returns:
            dict: Dictionary mapping item ID to array of top similar item IDs
                  Example: {Matrix: [MatrixReloaded, Inception, ...]}
        """
        top_n = top_n or Config.TOP_N_SIMILAR
        
        print(f"Finding top {top_n} similar items for each item...")
        start = datetime.now()
        
        item_ids = np.unique(similarity_matrix.nonzero()[1])
        similar_items = {}
        
        for item in item_ids:
            # Get all similarities for this item
            sim_items = similarity_matrix[item].toarray().ravel().argsort()[::-1][1:]
            # Store top N (excluding self at position 0)
            similar_items[item] = sim_items[:top_n]
        
        print(f"Processed {len(similar_items)} items")
        print(f"Time taken: {datetime.now() - start}\n")
        
        return similar_items
    
    @staticmethod
    def get_similar_items_for_item(similarity_matrix, item_id, top_n=10):
        """
        Get top N similar items for a specific item
        
        Args:
            similarity_matrix: Similarity matrix
            item_id: ID of the item
            top_n: Number of similar items to return
            
        Returns:
            tuple: (similar_item_ids, similarity_scores)
        """
        similarities = similarity_matrix[item_id].toarray().ravel()
        # Sort in descending order and get indices (exclude self)
        sim_indices = similarities.argsort()[::-1][1:top_n+1]
        sim_scores = similarities[sim_indices]
        
        return sim_indices, sim_scores
    
    @staticmethod
    def print_similar_movies(movie_id, similarity_matrix, movie_titles, top_n=10):
        """
        Print similar movies for a given movie
        
        Args:
            movie_id: ID of the movie
            similarity_matrix: Movie similarity matrix
            movie_titles: DataFrame with movie titles
            top_n: Number of similar movies to display
        """
        similar_ids, scores = SimilarityComputer.get_similar_items_for_item(
            similarity_matrix, movie_id, top_n
        )
        
        print(f"\nMovie: {movie_titles.loc[movie_id]['title']}")
        print(f"Year: {movie_titles.loc[movie_id]['year_of_release']}")
        print(f"\nTop {top_n} Similar Movies:")
        print("-" * 70)
        
        for i, (sim_id, score) in enumerate(zip(similar_ids, scores), 1):
            if sim_id in movie_titles.index:
                title = movie_titles.loc[sim_id]['title']
                year = movie_titles.loc[sim_id]['year_of_release']
                print(f"{i:2d}. {title} ({year}) - Similarity: {score:.4f}")
        
        print("-" * 70 + "\n")
