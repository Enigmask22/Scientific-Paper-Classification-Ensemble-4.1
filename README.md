## Project Title
Phân loại Chủ đề Bài báo Khoa học bằng Kỹ thuật Ensemble Learning

## Short Description
This project builds a robust text classification pipeline to predict the scientific topic of an arXiv abstract using ensemble learning methods. It compares classic vectorizers (Bag-of-Words, TF-IDF) and modern sentence embeddings with multiple ensemble classifiers to identify the most effective combination.

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Hugging Face Datasets
- Matplotlib / Seaborn
- Sentence-Transformers (for embeddings)
- PyTorch (dependency of sentence-transformers)

## Problem Statement
Given an arXiv abstract, predict its primary scientific category. The dataset includes abstracts labeled with categories like `astro-ph`, `cond-mat`, `cs`, `math`, and `physics`. The challenge is to evaluate which text representation and ensemble method produce the best classification performance.

## Methodology
1. Data Loading
   - Load dataset: `UniverseTBD/arxiv-abstracts-large` from Hugging Face Datasets.
   - Cache directory: `./cache`.

2. Sampling
   - Select the first 1,000 valid samples where the paper has exactly one category and that category’s top-level prefix is in: `['astro-ph', 'cond-mat', 'cs', 'math', 'physics']`.

3. Preprocessing
   - Strip and normalize whitespace.
   - Remove line breaks, special characters, and digits.
   - Lowercase text.
   - Map string labels to integer IDs.

4. Train/Test Split
   - Stratified split with 80% train and 20% test.

5. Text Vectorization
   - Bag-of-Words with `CountVectorizer`.
   - TF-IDF with `TfidfVectorizer`.
   - Sentence embeddings using `SentenceTransformer('intfloat/multilingual-e5-base')`.
     - Queries formatted as `query: <text>` to leverage model’s instruction-tuned format.
     - Normalize embeddings.

6. Models (Ensemble Learners)
   - Random Forest
   - AdaBoost
   - Gradient Boosting (sklearn)
   - XGBoost
   - LightGBM

7. Evaluation
   - Compute accuracy and classification report for each model–vectorizer pair.
   - Compare across BoW, TF-IDF, and Embeddings.

## Results
Below is a template table to record the achieved accuracies (fill after running).

| Model                | BoW (Accuracy) | TF-IDF (Accuracy) | Embeddings (Accuracy) |
|---------------------:|:--------------:|:-----------------:|:---------------------:|
| Random Forest        |     0.7950     |       0.7750      |        0.8550         |
| AdaBoost             |     0.6550     |       0.6700      |        0.7950         |
| Gradient Boosting    |     0.8050     |       0.7950      |        0.8650         |
| XGBoost              |     0.7850     |       0.7650      |        0.8750         |
| LightGBM             |     0.7950     |       0.8000      |        0.8750         |

Tips:
- Embeddings often help on semantic tasks, but tree models on dense embeddings may need parameter tuning (e.g., fewer estimators, learning rate).
- BoW/TF-IDF + tree ensembles can be strong baselines; try limiting features or using `max_features` to control dimensionality.

## How to Run

### 1) Environment Setup
- Python 3.10+ is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

If PyTorch/LightGBM build fails on Windows:
- Install Microsoft C++ Build Tools.
- For PyTorch, consider the official site to get a wheel matching your CUDA/CPU setup.

### 2) Download Data and Run
Run the main script:
```bash
python main.py
```

What it does:
- Downloads and caches the Hugging Face dataset.
- Samples and preprocesses 1,000 abstracts.
- Builds BoW, TF-IDF, and embeddings.
- Trains RandomForest, AdaBoost, GradientBoosting, XGBoost, and LightGBM.
- Prints accuracy for each configuration.

### 3) Reproducibility and Customization
- Change categories via `CATEGORIES_TO_SELECT` in `main.py`.
- Adjust the sample size by the stopping condition inside the sampling loop.
- Switch embedding model in `EmbeddingVectorizer(model_name=...)`.
- Tweak learners’ hyperparameters (e.g., `n_estimators`) in helper functions.

### Notes
- The notebook `Topic_Modeling_SHAP.ipynb` can be used to extend analysis and explanations (e.g., SHAP-based interpretation for classic models).
- Ensure adequate memory; embeddings computation can be resource intensive.