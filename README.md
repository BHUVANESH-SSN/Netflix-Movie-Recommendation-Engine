# Netflix Movie Recommendation Engine

A comprehensive machine learning project that implements a movie recommendation system using the Netflix Prize dataset. This project explores various collaborative filtering techniques, similarity-based methods, and ensemble approaches to predict user ratings for movies.

## 📋 Table of Contents

- [Problem Description](#problem-description)
- [Problem Statement](#problem-statement)
- [Business Objectives and Constraints](#business-objectives-and-constraints)
- [Machine Learning Problem](#machine-learning-problem)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Author](#author)

## 🎯 Problem Description

Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: **CinematchSM**. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix uses those predictions to make personal movie recommendations based on each customer's unique tastes.

This project explores alternative approaches to improve upon Cinematch's predictions by making better recommendations that could make a big difference to customers and business.

## 📝 Problem Statement

Netflix provided anonymous rating data with a prediction accuracy bar that is **10% better than what Cinematch can do** on the same training data set. The goal is to predict ratings that closely match subsequent actual ratings.

## 🎯 Business Objectives and Constraints

### Objectives:
1. Predict the rating that a user would give to a movie that they have not yet rated
2. Minimize the difference between predicted and actual rating (RMSE and MAPE)

### Constraints:
1. Some form of interpretability is required
2. Real-time prediction capabilities

## 🤖 Machine Learning Problem

### Type of Problem
- **Regression Problem**: Predicting continuous rating values (1-5 stars)
- **Collaborative Filtering**: Using user-movie interactions to make predictions

### Performance Metrics
- **RMSE (Root Mean Square Error)**: Primary metric for evaluation
- **MAPE (Mean Absolute Percentage Error)**: Secondary metric
- Goal: Minimize prediction error

## 📊 Data Sources

- **Primary Source**: [Netflix Prize Dataset on Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- **Official Netflix Prize**: [Netflix Prize Rules](https://www.netflixprize.com/rules.html)
- **Research Paper**: [Matrix Factorization Techniques](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf)

### Data Overview
The dataset contains:
- Anonymous rating data from Netflix users
- Movie ratings on a scale of 1-5
- Timestamps for each rating
- User IDs and Movie IDs

### Dataset Files

The Netflix Prize dataset consists of 5 main files:

1. **movie_titles.csv** - Contains movie information
   - Movie ID
   - Year of release
   - Movie title

2. **combined_data_1.txt** - Training data (Part 1)
   - Contains movie ratings in the format:
   - MovieID:
   - CustomerID, Rating, Date

3. **combined_data_2.txt** - Training data (Part 2)
   - Same format as combined_data_1.txt
   - Contains additional movie ratings

4. **combined_data_3.txt** - Training data (Part 3)
   - Same format as combined_data_1.txt
   - Contains additional movie ratings

5. **combined_data_4.txt** - Training data (Part 4)
   - Same format as combined_data_1.txt
   - Contains additional movie ratings

**Total Dataset Size:**
- Over 100 million ratings
- From 480,000+ users
- On 17,000+ movies
- Ratings span from October 1998 to December 2005

**Data Format Example:**
```
1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
30878,4,2005-12-26
...
```

The four `combined_data_*.txt` files are used for training and testing the recommendation models. These files are split into train (80%) and test (20%) sets during the preprocessing phase.

## 📁 Project Structure

```
Netflix_Movie_Recommendation_Engine/
│
├── netflix_movie_recommendation.ipynb    # Main Jupyter notebook
├── README.md                              # Project documentation
├── .gitignore                             # Git ignore file
│
└── data_folder/                           # Dataset directory (not tracked in git)
    ├── movie_titles.csv                   # Movie information
    ├── combined_data_1.txt                # Training data part 1
    ├── combined_data_2.txt                # Training data part 2
    ├── combined_data_3.txt                # Training data part 3
    ├── combined_data_4.txt                # Training data part 4
    ├── probe.txt                          # Probe/validation set
    └── qualifying.txt                     # Qualifying/test set
```

## 🔬 Machine Learning Models

This project implements and compares multiple recommendation algorithms:

### Baseline Models
- Global Average Rating
- User-based Average
- Movie-based Average

### Collaborative Filtering Models
- **User-User Similarity**: Based on user rating patterns
- **Item-Item Similarity**: Based on movie rating patterns
- **Cosine Similarity**: Measuring similarity between users/items

### Advanced Models (using Surprise library)
- **SVD (Singular Value Decomposition)**: Matrix factorization technique
- **SVD++**: Enhanced SVD with implicit feedback
- **KNN Basic**: K-Nearest Neighbors baseline
- **KNN with Means**: KNN with mean-centered ratings
- **KNN Baseline**: KNN with baseline estimates
- **Matrix Factorization**: Latent factor models

### Ensemble Methods
- Combining multiple models for better predictions

## 📈 Performance Metrics

Models are evaluated using:
- **RMSE (Root Mean Square Error)**
- **MAPE (Mean Absolute Percentage Error)**
- Train-Test Split: 80-20

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn
pip install scipy scikit-learn
pip install scikit-surprise
```

### Installing Surprise Library
```bash
pip install scikit-surprise
```

Or from source:
```bash
git clone https://github.com/NicolasHug/Surprise.git
cd Surprise
pip install .
```

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/BHUVANESH-SSN/Netflix-Movie-Recommendation-Engine.git
cd Netflix_Movie_Recommendation_Engine
```

2. Download the Netflix Prize dataset from [Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data)

3. Open the Jupyter notebook:
```bash
jupyter notebook netflix_movie_recommendation.ipynb
```

4. Run the cells sequentially to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Train various models
   - Compare model performance
   - Generate predictions

## 💻 Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities
- **Surprise**: Recommender system library
- **SciPy**: Scientific computing (sparse matrices, similarity computations)
- **Jupyter Notebook**: Interactive development environment

## 🔍 Key Features

- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Handling of cold start problems
- ✅ Multiple similarity metrics (Cosine, Pearson correlation)
- ✅ Sparse matrix optimization for large datasets
- ✅ Cross-validation for model evaluation
- ✅ Comparison of multiple recommendation algorithms
- ✅ Visualization of rating distributions and patterns

---

## 📊 Complete Project Flow & Methodology

### **Phase 1: Data Preprocessing & Exploration**

#### **Step 1.1: Data Merging**
**Problem:** Netflix data is split across 4 large text files (`combined_data_1.txt` to `combined_data_4.txt`)

**Solution:**
- **Library:** `pandas`, native Python file I/O
- **Technique:** Sequential file reading and concatenation
- **Process:**
  1. Read each file line by line
  2. Parse movie IDs (lines ending with `:`)
  3. Parse ratings (format: `CustomerID, Rating, Date`)
  4. Merge all into single CSV file
- **Output:** `data.csv` with columns: `movie`, `user`, `rating`, `date`

```python
# Format transformation
# From: MovieID:
#       CustomerID, Rating, Date
# To:   movie, user, rating, date
```

#### **Step 1.2: Data Quality Analysis**
**Algorithms Used:**
- **Duplicate Detection:** Hash-based duplicate checking on (`movie`, `user`, `rating`)
- **Missing Value Analysis:** Boolean masking with `isnull()`
- **Statistical Summary:** Descriptive statistics (mean, std, quartiles)

**Libraries:** `pandas`, `numpy`

**Metrics Computed:**
- Total ratings: 100+ million
- Unique users: 480,189
- Unique movies: 17,770
- Date range: Oct 1998 - Dec 2005

#### **Step 1.3: Train-Test Split**
**Technique:** Time-based split (preserves temporal order)
- **Split Ratio:** 80% Train, 20% Test
- **Method:** Chronological splitting (not random)
- **Reason:** Mimics real-world scenario where we predict future ratings

**Libraries:** `pandas`

---

### **Phase 2: Exploratory Data Analysis (EDA)**

#### **Step 2.1: Rating Distribution Analysis**
**Visualization Libraries:** `matplotlib`, `seaborn`

**Techniques:**
- **Count Plots:** Frequency of each rating (1-5 stars)
- **KDE (Kernel Density Estimation):** Smooth probability distributions
- **CDF (Cumulative Distribution Function):** Cumulative probability curves

**Key Findings:**
- Rating 4 is most common
- Distribution is left-skewed (more high ratings)
- Very few ratings of 1

#### **Step 2.2: Temporal Analysis**
**Technique:** Time series resampling
- **Method:** Monthly aggregation using `resample('M')`
- **Library:** `pandas` datetime operations

**Insights:**
- Peak rating periods identified
- Seasonal patterns in user behavior
- Growth trends over time

#### **Step 2.3: User Behavior Analysis**
**Statistical Techniques:**
- **Quantile Analysis:** Percentile-based user segmentation
- **Distribution Analysis:** PDF and CDF plots

**Key Metrics:**
- Average ratings per user
- Power users (top 5% most active)
- Distribution of user activity levels

**Findings:**
- Most users rate few movies
- Small percentage of "power users" contribute majority of ratings
- Long-tail distribution

#### **Step 2.4: Movie Popularity Analysis**
**Metrics:**
- Ratings per movie
- Average rating per movie
- Popularity distribution

**Visualization:** Histogram, box plots

---

### **Phase 3: Sparse Matrix Creation**

#### **Problem:** 
- 480K users × 17K movies = 8.16 billion possible entries
- Only 100M actual ratings (~1.2% density)
- Memory: Dense matrix would require ~60GB RAM

#### **Solution: Sparse Matrix Representation**

**Library:** `scipy.sparse`

**Data Structure:** CSR (Compressed Sparse Row) Matrix
- **Why CSR?** Efficient row slicing (user-based operations)
- **Alternative:** CSC for column operations (movie-based)

**Mathematical Representation:**
```
R[u,m] = rating if user u rated movie m, else 0
Shape: (users × movies)
Storage: Only non-zero values + indices
```

**Code Flow:**
```python
sparse_matrix = csr_matrix(
    (ratings, (user_indices, movie_indices)),
    shape=(n_users, n_movies)
)
```

**Benefits:**
- Memory: ~2GB instead of 60GB
- Fast row/column access
- Efficient matrix operations

**Sparsity Calculation:**
```
Sparsity = (1 - non_zero_count / total_cells) × 100
Result: ~98.8% sparse
```

---

### **Phase 4: Feature Engineering**

#### **Step 4.1: Global Average Rating**
**Formula:** 
```
μ = Σ(all ratings) / total_count
```

**Purpose:** Baseline prediction for cold start problems

#### **Step 4.2: User Average Ratings**
**Algorithm:**
```python
for each user u:
    user_avg[u] = mean(ratings by user u)
```

**Library:** `numpy` array operations

**Use Case:** 
- User bias estimation
- Handling new movies (user's typical rating)

#### **Step 4.3: Movie Average Ratings**
**Algorithm:**
```python
for each movie m:
    movie_avg[m] = mean(ratings for movie m)
```

**Use Case:**
- Movie popularity indicator
- Handling new users (movie's typical rating)

#### **Step 4.4: Cold Start Analysis**

**Cold Start Problem:** How to recommend when:
1. **New User:** No rating history
2. **New Movie:** No ratings received

**Solution Strategy:**
- **New Users:** Use movie average ratings
- **New Movies:** Use user average ratings
- **Both New:** Use global average

**Statistics Computed:**
- % of test users not in training: ~15.6%
- % of test movies not in training: ~2%

---

### **Phase 5: Similarity Computation**

#### **Step 5.1: Movie-Movie Similarity**

**Algorithm:** Cosine Similarity

**Mathematical Formula:**
```
similarity(m_i, m_j) = (R_i · R_j) / (||R_i|| × ||R_j||)

Where:
- R_i = rating vector for movie i
- R_j = rating vector for movie j
- · = dot product
- || || = L2 norm (Euclidean length)
```

**Library:** `sklearn.metrics.pairwise.cosine_similarity`

**Why Cosine Similarity?**
- Scale-invariant (handles rating magnitude differences)
- Efficient for sparse matrices
- Value range: [-1, 1] (1 = identical, 0 = no similarity)

**Computational Complexity:**
- Time: O(n² × m) for n movies, m users
- Space: O(n²) but sparse storage used

**Process:**
```python
# Transpose matrix: movies become rows
movie_matrix = user_movie_matrix.T

# Compute pairwise similarities
movie_similarity = cosine_similarity(movie_matrix, dense_output=False)
```

**Optimization:** Store only top-100 similar movies per movie

#### **Step 5.2: User-User Similarity (Attempted)**

**Challenge:** 
- 480K users → 480K × 480K similarity matrix
- Computational time: ~42 days on single core
- Memory: ~920GB uncompressed

**Attempted Solution:** Dimensionality Reduction

**Technique:** TruncatedSVD (Singular Value Decomposition)

**Mathematical Background:**
```
R ≈ U × Σ × V^T

Where:
- R = original rating matrix
- U = user factors (users × k)
- Σ = diagonal matrix of singular values
- V^T = movie factors (k × movies)
- k = number of latent factors (500 in our case)
```

**Library:** `sklearn.decomposition.TruncatedSVD`

**Process:**
1. Reduce 17K dimensions to 500 latent factors
2. Compute similarity in reduced space
3. Expected speedup: ~34x

**Result:** Still too slow (dense matrix issue)

**Final Approach:** Runtime computation
- Compute similarities on-demand for specific users
- Cache computed similarities
- Dictionary-based storage

---

### **Phase 6: Sampling Strategy**

#### **Why Sample?**
- Full dataset: 100M+ ratings
- Training time: Days to weeks
- Development: Need quick iterations

#### **Sampling Technique:**

**Method:** Stratified Random Sampling
- **Library:** `numpy.random`
- **Seed:** 15 (reproducibility)

**Sample Sizes:**
- **Training:** 10,000 users, 1,000 movies
- **Testing:** 5,000 users, 500 movies

**Algorithm:**
```python
# Fixed seed for reproducibility
np.random.seed(15)

# Random selection without replacement
sample_users = np.random.choice(all_users, n_users, replace=False)
sample_movies = np.random.choice(all_movies, n_movies, replace=False)

# Filter ratings matrix
mask = (users in sample_users) & (movies in sample_movies)
sample_matrix = original_matrix[mask]
```

**Validation:** Ensures representativeness
- Rating distribution preserved
- User/movie activity patterns maintained

---

### **Phase 7: Advanced Feature Engineering for ML Models**

#### **Feature Set Design**

For each (user, movie) pair, we create 13 baseline features:

**1. Global Features (1 feature):**
- `GAvg`: Global average rating (μ)

**2. Similar Users' Ratings (5 features):**
- `sur1` to `sur5`: Ratings from top 5 similar users for this movie

**Algorithm:**
```python
# Find similar users
user_similarity = cosine_similarity(current_user_vector, all_users)
top_similar_users = argsort(user_similarity)[-5:]

# Get their ratings for this movie
for similar_user in top_similar_users:
    feature = rating[similar_user, current_movie]
```

**Fallback:** If fewer than 5, pad with movie average

**3. Similar Movies' Ratings (5 features):**
- `smr1` to `smr5`: Current user's ratings for top 5 similar movies

**Algorithm:**
```python
# Find similar movies
movie_similarity = cosine_similarity(current_movie_vector, all_movies)
top_similar_movies = argsort(movie_similarity)[-5:]

# Get user's ratings for these movies
for similar_movie in top_similar_movies:
    feature = rating[current_user, similar_movie]
```

**Fallback:** Pad with user average

**4. User Average (1 feature):**
- `UAvg`: Average rating given by this user

**5. Movie Average (1 feature):**
- `MAvg`: Average rating received by this movie

**Feature Vector:** `[GAvg, sur1-5, smr1-5, UAvg, MAvg]` (13 features)

**Computational Challenge:**
- Creating features: ~26 hours for full training data
- Solution: Parallel processing, caching, pre-computation

---

### **Phase 8: Machine Learning Models**

#### **Model 1: Baseline Model (Surprise Library)**

**Algorithm:** Bias-based prediction

**Mathematical Formula:**
```
predicted_rating = μ + b_u + b_i

Where:
- μ = global average
- b_u = user bias
- b_i = item (movie) bias
```

**Optimization:** Stochastic Gradient Descent (SGD)

**Loss Function (Ridge Regression):**
```
L = Σ(r_ui - μ - b_u - b_i)² + λ(b_u² + b_i²)
```

**Library:** `surprise.BaselineOnly`

**Hyperparameters:**
- Method: SGD
- Learning rate: 0.001
- Regularization (λ): Default

**Performance:**
- **RMSE:** ~1.05
- **MAPE:** ~22%

---

#### **Model 2: KNN Baseline (User-User)**

**Algorithm:** K-Nearest Neighbors with Baseline

**Prediction Formula:**
```
r̂_ui = b_ui + (Σ_v∈N_k(u) sim(u,v) × (r_vi - b_vi)) / (Σ_v∈N_k(u) sim(u,v))

Where:
- b_ui = baseline prediction (μ + b_u + b_i)
- N_k(u) = k nearest neighbors of user u who rated item i
- sim(u,v) = similarity between users u and v
```

**Similarity Metric:** Pearson Baseline Correlation

**Formula:**
```
sim(u,v) = Σ_i (r_ui - b_ui)(r_vi - b_vi) / sqrt(Σ(r_ui - b_ui)² × Σ(r_vi - b_vi)²)
```

**Shrinkage:** Applied to handle users with few common ratings

**Library:** `surprise.KNNBaseline`

**Hyperparameters:**
- k = 40 neighbors
- Similarity: Pearson Baseline
- Shrinkage: 100
- Min support: 2 common ratings

**Performance:**
- **RMSE:** ~1.02
- **MAPE:** ~21%

---

#### **Model 3: KNN Baseline (Item-Item)**

**Same as User-User but:**
- Computes movie-movie similarities
- Finds k similar movies rated by user
- Generally faster (fewer movies than users)

**Prediction Formula:**
```
r̂_ui = b_ui + (Σ_j∈N_k(i) sim(i,j) × (r_uj - b_uj)) / (Σ_j∈N_k(i) sim(i,j))
```

**Performance:**
- **RMSE:** ~1.01
- **MAPE:** ~20.5%

---

#### **Model 4: SVD (Matrix Factorization)**

**Algorithm:** Singular Value Decomposition

**Mathematical Concept:**
```
R ≈ P × Q^T

Where:
- R = rating matrix (users × movies)
- P = user factor matrix (users × k)
- Q = movie factor matrix (movies × k)
- k = number of latent factors
```

**Prediction Formula:**
```
r̂_ui = μ + b_u + b_i + q_i^T × p_u

Where:
- q_i = latent factor vector for movie i
- p_u = latent factor vector for user u
```

**Optimization:** Alternating Least Squares (ALS) or SGD

**Loss Function:**
```
L = Σ(r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
```

**Library:** `surprise.SVD`

**Hyperparameters:**
- n_factors = 100 (latent dimensions)
- Biased: True
- Random state: 15

**Intuition:**
- Discovers hidden patterns (genres, themes)
- Example factors: action level, comedy level, drama level
- Users/movies mapped to factor space

**Performance:**
- **RMSE:** ~0.98
- **MAPE:** ~19%

---

#### **Model 5: SVD++ (Enhanced Matrix Factorization)**

**Enhancement:** Incorporates implicit feedback

**Prediction Formula:**
```
r̂_ui = μ + b_u + b_i + q_i^T × (p_u + |I_u|^(-0.5) × Σ_j∈I_u y_j)

Where:
- I_u = set of movies rated by user u
- y_j = implicit factor for movie j
- |I_u|^(-0.5) = normalization term
```

**Key Idea:**
- A user's preferences influenced by ALL movies they rated
- Even unrated movies provide information (implicit feedback)
- More comprehensive user representation

**Library:** `surprise.SVDpp`

**Hyperparameters:**
- n_factors = 50
- Random state: 15

**Computational Cost:** Higher than SVD (more parameters)

**Performance:**
- **RMSE:** ~0.96
- **MAPE:** ~18.5%

---

#### **Model 6: XGBoost (Gradient Boosting)**

**Algorithm:** Extreme Gradient Boosting

**Concept:** Ensemble of decision trees

**Mathematical Framework:**
```
ŷ_i = Σ_k f_k(x_i)

Where:
- f_k = k-th decision tree
- Each tree corrects errors of previous trees
```

**Objective Function:**
```
Obj = Σ_i L(y_i, ŷ_i) + Σ_k Ω(f_k)

Where:
- L = loss function (MSE for regression)
- Ω = regularization term (prevents overfitting)
```

**Library:** `xgboost`

**Hyperparameter Tuning:** RandomizedSearchCV

**Search Space:**
- `learning_rate`: [0.01, 0.2]
- `n_estimators`: [100, 1000]
- `max_depth`: [1, 10]
- `min_child_weight`: [1, 8]
- `gamma`: [0, 0.02]
- `subsample`: [0.6, 1.0]
- `reg_alpha`: [0, 200] (L1 regularization)
- `reg_lambda`: [0, 200] (L2 regularization)
- `colsample_bytree`: [0.6, 0.9]

**Tuning Method:**
- **Algorithm:** Randomized Search
- **CV Folds:** 3
- **Iterations:** 10
- **Scoring:** Negative MSE

**Feature Importance:**
- XGBoost provides feature importance scores
- Identifies most predictive features
- Typically: Baseline predictions > Averages > Similar users/movies

---

#### **Model 7: XGBoost Ensemble (Final Model)**

**Strategy:** Stacking/Blending

**Concept:** Use predictions from other models as features

**Feature Set (17 features):**
1. Original 13 features (GAvg, sur1-5, smr1-5, UAvg, MAvg)
2. `bslpr`: Baseline model prediction
3. `knn_bsl_u`: KNN User-User prediction
4. `knn_bsl_m`: KNN Item-Item prediction
5. `svd`: SVD prediction
6. `svdpp`: SVD++ prediction

**Architecture:**
```
Level 0 Models (Base Models):
├── Baseline
├── KNN User-User
├── KNN Item-Item
├── SVD
└── SVD++

Level 1 Model (Meta-Model):
└── XGBoost (combines all predictions)
```

**Training Process:**
1. Train all base models on training data
2. Generate predictions from each model
3. Use predictions as features for XGBoost
4. Train XGBoost on augmented feature set

**Advantages:**
- Captures strengths of each model
- Reduces individual model weaknesses
- Often achieves best performance

**Performance:**
- **RMSE:** ~0.93-0.95 (best overall)
- **MAPE:** ~17-18%

---

### **Phase 9: Model Evaluation & Comparison**

#### **Evaluation Metrics**

**1. RMSE (Root Mean Square Error)**
```
RMSE = sqrt(Σ(predicted - actual)² / n)
```

**Interpretation:**
- Average prediction error in rating points
- Lower is better
- Penalizes large errors more (squared term)

**2. MAPE (Mean Absolute Percentage Error)**
```
MAPE = (Σ|predicted - actual| / actual) / n × 100
```

**Interpretation:**
- Average percentage error
- Scale-independent
- Easy to interpret (e.g., 20% error)

#### **Cross-Validation**
- **Method:** 3-fold CV
- **Purpose:** Prevent overfitting
- **Library:** `sklearn.model_selection`

#### **Visualization Techniques**

**1. Bar Charts:** Compare RMSE/MAPE across models

**2. Train vs Test:** Identify overfitting
```
If train_error << test_error → Overfitting
```

**3. Feature Importance Plots:**
- XGBoost provides importance scores
- Identifies key predictive features

---

### **Phase 10: Production Considerations**

#### **Scaling Strategies**

**1. Distributed Computing:**
- **Framework:** Apache Spark
- **Library:** PySpark MLlib
- **Benefit:** Handle full dataset

**2. Approximate Methods:**
- **Locality Sensitive Hashing (LSH):** Fast similarity search
- **Matrix sketching:** Dimensionality reduction

**3. Incremental Learning:**
- **Online learning:** Update models with new ratings
- **Warm start:** Continue training from saved state

#### **Cold Start Solutions**

**1. Content-Based Filtering:**
- Use movie metadata (genre, actors, director)
- Don't rely solely on collaborative filtering

**2. Hybrid Approaches:**
- Combine collaborative + content-based
- Use demographics for new users

**3. Popularity-Based:**
- Recommend trending movies to new users
- Safe fallback strategy

---

## 🎯 Key Takeaways

### **Technical Insights:**

1. **Sparse Matrices Are Essential**
   - 98.8% sparsity requires specialized data structures
   - CSR format optimal for row operations

2. **Similarity Computation Is Expensive**
   - User-user similarity infeasible for large datasets
   - Item-item similarity more practical (fewer items)

3. **Feature Engineering Matters**
   - Domain knowledge improves predictions
   - Baseline predictions crucial for handling cold start

4. **Ensemble Methods Win**
   - Combining multiple models beats individual models
   - XGBoost excellent for blending predictions

5. **Implicit Feedback Helps**
   - SVD++ outperforms SVD
   - User's rating history provides valuable signal

### **Performance Summary:**

| Model | RMSE | MAPE | Complexity |
|-------|------|------|------------|
| Baseline | 1.05 | 22% | Low |
| KNN User-User | 1.02 | 21% | Medium |
| KNN Item-Item | 1.01 | 20.5% | Medium |
| SVD | 0.98 | 19% | Medium-High |
| SVD++ | 0.96 | 18.5% | High |
| XGBoost | 0.95 | 18% | Medium |
| **XGBoost Ensemble** | **0.93** | **17%** | **High** |

### **Libraries Used:**

| Purpose | Libraries |
|---------|-----------|
| Data Processing | `pandas`, `numpy` |
| Sparse Matrices | `scipy.sparse` |
| Similarity | `sklearn.metrics.pairwise` |
| Collaborative Filtering | `surprise` |
| Gradient Boosting | `xgboost` |
| Visualization | `matplotlib`, `seaborn` |
| ML Utilities | `sklearn` |

### **Algorithms Implemented:**

1. ✅ **Cosine Similarity** (Movie-Movie, User-User)
2. ✅ **Pearson Correlation** (Baseline similarity)
3. ✅ **Baseline Prediction** (Bias estimation)
4. ✅ **K-Nearest Neighbors** (Memory-based CF)
5. ✅ **SVD** (Matrix factorization)
6. ✅ **SVD++** (Enhanced matrix factorization)
7. ✅ **XGBoost** (Gradient boosted trees)
8. ✅ **Ensemble Stacking** (Meta-learning)

---

---

## 📖 Function Reference Guide

This section provides a comprehensive overview of all functions in the project, their purpose, and how to use them.

### **Data Preprocessing Functions** (`src/data_preprocessing.py`)

#### **1. merge_netflix_data()**
**Purpose:** Combines 4 separate Netflix data files into one CSV
```python
from src.data_preprocessing import DataPreprocessor

# Merge all Netflix data files
DataPreprocessor.merge_netflix_data()
```
**Use Case:** Run once at the beginning to prepare data  
**Output:** Creates `netflix_data.csv` with all ratings

---

#### **2. load_and_sort_data()**
**Purpose:** Loads merged CSV and sorts by date (oldest to newest)
```python
# Load and sort data by date
df = DataPreprocessor.load_and_sort_data('netflix_data.csv')
print(f"Loaded {len(df)} ratings")
```
**Use Case:** After merging, load data for processing  
**Why Sort?** Ensures train data is "past" and test data is "future"

---

#### **3. check_data_quality()**
**Purpose:** Inspects data for missing values, duplicates, and basic stats
```python
# Check data quality
quality_metrics = DataPreprocessor.check_data_quality(df)
print(f"Missing values: {quality_metrics['nan_count']}")
print(f"Duplicates: {quality_metrics['duplicates']}")
```
**Use Case:** Quality assurance before training models  
**Returns:** Dictionary with counts of users, movies, ratings, NaNs, duplicates

---

#### **4. split_train_test()**
**Purpose:** Splits data into training (80%) and testing (20%) sets
```python
# Split into train and test
train_df, test_df = DataPreprocessor.split_train_test(df, train_ratio=0.80)
print(f"Train: {len(train_df)} ratings")
print(f"Test: {len(test_df)} ratings")
```
**Use Case:** Create separate datasets for training and evaluation  
**Important:** Split is time-based (first 80% = train, last 20% = test)

---

### **Sparse Matrix Functions** (`src/sparse_matrix_handler.py`)

#### **5. create_sparse_matrix()**
**Purpose:** Converts dataframe to memory-efficient sparse matrix format
```python
from src.sparse_matrix_handler import SparseMatrixHandler

# Create sparse matrix
sparse_matrix = SparseMatrixHandler.create_sparse_matrix(
    train_df, 
    filename='train_sparse.npz'
)
print(f"Matrix shape: {sparse_matrix.shape}")  # (users, movies)
```
**Use Case:** Convert ratings into matrix for mathematical operations  
**Why Sparse?** Saves memory (2GB instead of 60GB for Netflix data)  
**Output:** CSR matrix where rows=users, columns=movies, values=ratings

---

#### **6. calculate_sparsity()**
**Purpose:** Calculates what percentage of matrix is empty
```python
# Calculate sparsity
sparsity = SparseMatrixHandler.calculate_sparsity(sparse_matrix)
print(f"Sparsity: {sparsity:.2f}%")  # Expected: ~98.8%
```
**Use Case:** Understand data density  
**Interpretation:** High sparsity (>95%) justifies using sparse matrices

---

### **Feature Engineering Functions** (`src/feature_engineering.py`)

#### **7. get_average_ratings()**
**Purpose:** Computes average rating for each user OR each movie
```python
from src.feature_engineering import FeatureEngineer

# Get user averages (how harsh/generous each user is)
user_avg = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=True)
print(f"User 12345 average: {user_avg[12345]}")

# Get movie averages (how good/bad each movie is)
movie_avg = FeatureEngineer.get_average_ratings(sparse_matrix, of_users=False)
print(f"Movie 500 average: {movie_avg[500]}")
```
**Use Case:** Feature for prediction models  
**Insight:** Some users rate harshly (avg 2.5), others generously (avg 4.5)

---

#### **8. compute_global_average()**
**Purpose:** Calculates overall average of all ratings
```python
# Get global average rating
global_avg = FeatureEngineer.compute_global_average(sparse_matrix)
print(f"Global average: {global_avg:.2f} stars")  # Usually ~3.6
```
**Use Case:** Baseline prediction when no other info available  
**Example:** Predict global_avg for brand new users/movies

---

#### **9. create_feature_set()**
**Purpose:** Creates comprehensive feature set for ML models
```python
# Create all features
features = FeatureEngineer.create_feature_set(
    train_df, 
    sparse_matrix, 
    user_avg, 
    movie_avg, 
    global_avg
)
```
**Use Case:** Prepare input features for XGBoost training  
**Output:** Dataframe with user_avg, movie_avg, and other features

---

### **Similarity Computation Functions** (`src/similarity.py`)

#### **10. compute_movie_similarity()**
**Purpose:** Calculates similarity score between every pair of movies
```python
from src.similarity import SimilarityComputer

# Compute movie-movie similarity
movie_sim_matrix = SimilarityComputer.compute_movie_similarity(
    sparse_matrix, 
    filename='movie_similarity.npz'
)
```
**Use Case:** Find similar movies for recommendations  
**Method:** Cosine similarity (compares rating patterns)  
**Output:** Matrix where cell (i,j) = similarity between movie i and movie j

---

#### **11. get_top_similar_items()**
**Purpose:** For each movie, keep only top N most similar movies
```python
# Get top 100 similar movies for each movie
top_similar = SimilarityComputer.get_top_similar_items(
    movie_sim_matrix, 
    top_n=100
)

# Check similar movies to movie 500
similar_to_500 = top_similar[500]
print(f"Top similar movies: {similar_to_500[:5]}")
```
**Use Case:** Reduce memory by keeping only most relevant similarities  
**Why?** Don't need all 17K similarities, top 100 is enough

---

#### **12. print_similar_movies()**
**Purpose:** Display human-readable list of similar movies
```python
# Print top 10 movies similar to "The Matrix"
SimilarityComputer.print_similar_movies(
    movie_id=133093, 
    movie_titles_df=movie_titles, 
    top_similar_dict=top_similar, 
    top_n=10
)
```
**Use Case:** Validate similarity computation, demo recommendations  
**Output:** Prints movie titles and similarity scores

---

### **Model Training Functions** (`src/models.py`)

#### **13. get_error_metrics()**
**Purpose:** Calculates RMSE and MAPE to evaluate predictions
```python
from src.models import ModelTrainer

# Evaluate predictions
rmse, mape = ModelTrainer.get_error_metrics(
    y_true=actual_ratings, 
    y_pred=predicted_ratings
)
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
```
**Use Case:** Measure how accurate your predictions are  
**Lower = Better:** RMSE < 1.0 is good, MAPE < 20% is good

---

#### **14. train_xgboost()**
**Purpose:** Trains XGBoost gradient boosting model
```python
# Train XGBoost with hyperparameter tuning
model, train_results, test_results = ModelTrainer.train_xgboost(
    x_train=train_features,
    y_train=train_ratings,
    x_test=test_features,
    y_test=test_ratings,
    tune_params=True,  # Enable hyperparameter search
    n_iter=20  # Try 20 random combinations
)
```
**Use Case:** Train powerful gradient boosting model  
**Features:** Automatic hyperparameter tuning with RandomizedSearchCV

---

#### **15. train_baseline_model()**
**Purpose:** Trains simple baseline model (global avg + user/movie bias)
```python
from surprise import BaselineOnly

# Train baseline model
algo = BaselineOnly()
train_results, test_results = ModelTrainer.train_surprise_model(
    algo=algo,
    trainset=train_surprise,
    testset=test_surprise,
    model_name="Baseline"
)
```
**Use Case:** Quick baseline to compare against  
**Prediction Formula:** `μ + user_bias + movie_bias`

---

#### **16. train_knn_baseline()**
**Purpose:** Trains K-Nearest Neighbors collaborative filtering
```python
from surprise import KNNBaseline

# User-based KNN
algo = KNNBaseline(k=40, sim_options={'name': 'pearson_baseline', 'user_based': True})
train_results, test_results = ModelTrainer.train_surprise_model(
    algo=algo,
    trainset=train_surprise,
    testset=test_surprise,
    model_name="KNN User-User"
)
```
**Use Case:** Similarity-based recommendations  
**Options:** User-based or Item-based, various similarity metrics

---

#### **17. train_svd()**
**Purpose:** Trains SVD matrix factorization model
```python
from surprise import SVD

# Train SVD
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
train_results, test_results = ModelTrainer.train_surprise_model(
    algo=algo,
    trainset=train_surprise,
    testset=test_surprise,
    model_name="SVD"
)
```
**Use Case:** Discover latent factors (hidden patterns like genres)  
**Method:** Decomposes user-movie matrix into lower dimensions

---

#### **18. train_svdpp()**
**Purpose:** Trains SVD++ (enhanced SVD with implicit feedback)
```python
from surprise import SVDpp

# Train SVD++ (slower but more accurate)
algo = SVDpp(n_factors=20, n_epochs=20, lr_all=0.007, reg_all=0.02)
train_results, test_results = ModelTrainer.train_surprise_model(
    algo=algo,
    trainset=train_surprise,
    testset=test_surprise,
    model_name="SVD++"
)
```
**Use Case:** Better than SVD, considers which movies user rated  
**Tradeoff:** More accurate but slower training

---

#### **19. train_surprise_model()**
**Purpose:** Generic training function for any Surprise algorithm
```python
# Train any Surprise model
train_results, test_results = ModelTrainer.train_surprise_model(
    algo=your_algorithm,
    trainset=train_data,
    testset=test_data,
    model_name="Your Model",
    verbose=True
)
```
**Use Case:** Wrapper for training Baseline, KNN, SVD, SVD++  
**Output:** Returns train and test performance metrics

---

### **Visualization Functions** (`src/visualization.py`)

#### **20. plot_rating_distribution()**
**Purpose:** Creates bar chart of rating frequency (1-5 stars)
```python
from src.visualization import Visualizer

# Plot rating distribution
Visualizer.plot_rating_distribution(df, title='Netflix Ratings Distribution')
```
**Use Case:** Understand data before modeling  
**Insight:** Usually see more 4s and 5s (positivity bias)

---

#### **21. plot_model_comparison()**
**Purpose:** Bar chart comparing all models' performance
```python
# Compare all models
results = {
    'Baseline': {'rmse': 1.05, 'mape': 22},
    'SVD': {'rmse': 0.98, 'mape': 19},
    'XGBoost': {'rmse': 0.95, 'mape': 18}
}

Visualizer.plot_model_comparison(results, metric='rmse')
```
**Use Case:** Visualize which model performs best  
**Output:** Horizontal bar chart (shorter bar = better model)

---

#### **22. plot_train_test_comparison()**
**Purpose:** Shows train vs test performance to detect overfitting
```python
# Plot train vs test errors
Visualizer.plot_train_test_comparison(
    train_results={'SVD': 0.85, 'XGBoost': 0.80},
    test_results={'SVD': 0.98, 'XGBoost': 0.95},
    metric='rmse'
)
```
**Use Case:** Check if model is overfitting  
**Red Flag:** Large gap between train and test error

---

### **Complete Usage Example**

Here's how all functions work together in a typical workflow:

```python
# Step 1: Data Preprocessing
from src.data_preprocessing import DataPreprocessor

DataPreprocessor.merge_netflix_data()
df = DataPreprocessor.load_and_sort_data()
quality = DataPreprocessor.check_data_quality(df)
train_df, test_df = DataPreprocessor.split_train_test(df)

# Step 2: Create Sparse Matrices
from src.sparse_matrix_handler import SparseMatrixHandler

train_matrix = SparseMatrixHandler.create_sparse_matrix(train_df)
test_matrix = SparseMatrixHandler.create_sparse_matrix(test_df)
sparsity = SparseMatrixHandler.calculate_sparsity(train_matrix)

# Step 3: Feature Engineering
from src.feature_engineering import FeatureEngineer

user_avg = FeatureEngineer.get_average_ratings(train_matrix, of_users=True)
movie_avg = FeatureEngineer.get_average_ratings(train_matrix, of_users=False)
global_avg = FeatureEngineer.compute_global_average(train_matrix)

# Step 4: Compute Similarities
from src.similarity import SimilarityComputer

movie_sim = SimilarityComputer.compute_movie_similarity(train_matrix)
top_similar = SimilarityComputer.get_top_similar_items(movie_sim, top_n=100)

# Step 5: Train Models
from src.models import ModelTrainer
from surprise import SVD

algo = SVD()
train_results, test_results = ModelTrainer.train_surprise_model(
    algo, trainset, testset, "SVD"
)

# Step 6: Visualize Results
from src.visualization import Visualizer

Visualizer.plot_rating_distribution(train_df)
Visualizer.plot_model_comparison(all_results, metric='rmse')
```

---

## 📚 References

- [Netflix Recommendations: Beyond the 5 Stars](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
- [Surprise Library Documentation](http://surprise.readthedocs.io/en/stable/getting_started.html)
- [Matrix Factorization Techniques for Recommender Systems](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf)
- [SVD Decomposition Tutorial](https://www.youtube.com/watch?v=P5mlg91as1c)

## 👨‍💻 Author

**BHUVANESH SSN**  
Computer Science Student  
SSN College of Engineering

**GitHub**: [@BHUVANESH-SSN](https://github.com/BHUVANESH-SSN)  
**Repository**: [Netflix-Movie-Recommendation-Engine](https://github.com/BHUVANESH-SSN/Netflix-Movie-Recommendation-Engine)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Netflix for providing the dataset and inspiration
- The Surprise library developers for excellent recommendation tools
- Research community for collaborative filtering techniques

---

⭐ If you find this project helpful, please consider giving it a star!
