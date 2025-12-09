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
