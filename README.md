# 🧠 Scientific Paper Topic Classification - Ensemble Learning

<div align="center">

![Topic Classification](https://img.shields.io/badge/Topic_Classification-v1.0.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![AI Engine](https://img.shields.io/badge/AI_Engine-Ensemble_Learning-red?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-87.5%25-green?style=for-the-badge)

**Scientific paper topic classification using Ensemble Learning techniques**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🛠️ Technologies](#️-technologies) • [📊 Results](#-results)

</div>

## 🌟 Key Features

### 🤖 Ensemble Learning Pipeline
- **5 powerful algorithms** - Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM
- **3 vectorization methods** - Bag-of-Words, TF-IDF, Sentence Embeddings
- **High accuracy** - Up to 87.5% with XGBoost + Embeddings
- **Comprehensive comparison** - 15 combinations to find optimal approach

### 📊 Text Processing & Analysis
- **Smart preprocessing** - Remove noise, normalize text
- **Sentence Embeddings** - Using multilingual-e5-base model
- **Stratified sampling** - Ensure balanced data
- **Real-time evaluation** - Detailed metrics for each model

### 🎯 Scientific Categories
- **5 main domains** - astro-ph, cond-mat, cs, math, physics
- **arXiv dataset** - 1000+ high-quality abstracts
- **Multi-class classification** - Accurate topic classification

## 🛠️ Công nghệ

### 🐍 Python Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-FFB000?style=flat&logo=lightgbm&logoColor=white)

- **Python 3.10+** - Core programming language
- **Scikit-learn** - Machine learning framework
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Microsoft's gradient boosting
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization

### 🤖 AI & NLP
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FF6B6B?style=flat&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FF6B6B?style=flat&logo=sentence-transformers&logoColor=white)

- **Hugging Face Datasets** - arXiv abstracts dataset
- **Sentence Transformers** - Multilingual embeddings
- **PyTorch** - Deep learning backend
- **Multilingual E5** - State-of-the-art embedding model

## 🎯 Problem Statement

Given an arXiv abstract, predict its primary scientific category. The dataset includes abstracts labeled with categories like `astro-ph`, `cond-mat`, `cs`, `math`, and `physics`. The challenge is to evaluate which text representation and ensemble method produce the best classification performance.

## 🔬 Methodology

### 📊 Data Pipeline
1. **Data Loading**
   - Load dataset: `UniverseTBD/arxiv-abstracts-large` from Hugging Face Datasets
   - Cache directory: `./cache`

2. **Sampling Strategy**
   - Select the first 1,000 valid samples where the paper has exactly one category
   - Categories: `['astro-ph', 'cond-mat', 'cs', 'math', 'physics']`

3. **Text Preprocessing**
   - Strip and normalize whitespace
   - Remove line breaks, special characters, and digits
   - Lowercase text
   - Map string labels to integer IDs

4. **Train/Test Split**
   - Stratified split with 80% train and 20% test

### 🤖 Model Architecture
5. **Text Vectorization**
   - **Bag-of-Words** with `CountVectorizer`
   - **TF-IDF** with `TfidfVectorizer`
   - **Sentence Embeddings** using `SentenceTransformer('intfloat/multilingual-e5-base')`
     - Queries formatted as `query: <text>` to leverage model's instruction-tuned format
     - Normalize embeddings

6. **Ensemble Learners**
   - **Random Forest** - Bootstrap aggregating
   - **AdaBoost** - Adaptive boosting
   - **Gradient Boosting** - sklearn implementation
   - **XGBoost** - Extreme gradient boosting
   - **LightGBM** - Microsoft's gradient boosting

7. **Evaluation Metrics**
   - Compute accuracy and classification report for each model–vectorizer pair
   - Compare across BoW, TF-IDF, and Embeddings

## 📊 Kết quả

### 🏆 Performance Comparison

| Model                | BoW (Accuracy) | TF-IDF (Accuracy) | Embeddings (Accuracy) |
|---------------------:|:--------------:|:-----------------:|:---------------------:|
| **Random Forest**    |     79.50%     |       77.50%      |        **85.50%**     |
| **AdaBoost**         |     65.50%     |       67.00%      |        **79.50%**     |
| **Gradient Boosting**|     **80.50%** |       79.50%      |        **86.50%**     |
| **XGBoost**          |     78.50%     |       76.50%      |        **87.50%**     |
| **LightGBM**         |     79.50%     |       **80.00%**  |        **87.50%**     |

### 🎯 Key Insights
- **🏆 Best Performance**: XGBoost + Embeddings (87.5%)
- **📈 Embeddings Advantage**: Consistently outperform BoW/TF-IDF
- **🌳 Tree Models**: Gradient Boosting shows strong baseline performance
- **⚡ LightGBM**: Fastest training with competitive accuracy

### 💡 Optimization Tips
- **Embeddings**: Often help on semantic tasks, but tree models on dense embeddings may need parameter tuning
- **Feature Engineering**: Try limiting features or using `max_features` to control dimensionality
- **Hyperparameter Tuning**: Adjust `n_estimators`, `learning_rate` for better performance

## 🚀 Bắt đầu nhanh

### 📋 Yêu cầu hệ thống
- **Python 3.10+** với pip
- **4GB+ RAM** (khuyến nghị 8GB+ cho embeddings)
- **Internet connection** (cho Hugging Face datasets)

### ⚡ Cài đặt tự động

```bash
# 1. Clone repository
git clone <repository-url>
cd scientific-paper-classification

# 2. Tạo virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### 🔧 Troubleshooting

<details>
<summary><strong>🐍 Python Dependencies Issues</strong></summary>

```bash
# Nếu PyTorch/LightGBM build fails trên Windows:
# Cài đặt Microsoft C++ Build Tools
# Hoặc sử dụng pre-built wheels:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install lightgbm --only-binary=all
```
</details>

### 🏃‍♂️ Chạy ứng dụng

```bash
# Chạy pipeline hoàn chỉnh
python main.py
```

**Quá trình thực hiện:**
- 📥 Downloads và cache Hugging Face dataset
- 🔄 Samples và preprocesses 1,000 abstracts  
- 🧮 Builds BoW, TF-IDF, và embeddings
- 🤖 Trains RandomForest, AdaBoost, GradientBoosting, XGBoost, LightGBM
- 📊 Prints accuracy cho mỗi configuration

### 🎛️ Customization

```python
# Thay đổi categories trong main.py
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

# Điều chỉnh sample size
if len(samples) >= 1000:  # Thay đổi số lượng samples
    break

# Switch embedding model
embedding_vectorizer = EmbeddingVectorizer(model_name='intfloat/multilingual-e5-base')

# Tweak hyperparameters
def train_and_test_random_forest(..., n_estimators: int = 100):
```

### 📚 Extended Analysis
- **Jupyter Notebook**: `Topic_Modeling_SHAP.ipynb` cho SHAP-based interpretation
- **Memory Requirements**: Embeddings computation có thể tốn nhiều RAM
- **GPU Support**: Có thể sử dụng GPU cho sentence transformers

## 🧪 Testing

```bash
# Test individual components
python -c "import main; print('All imports successful')"

# Test data loading
python -c "from datasets import load_dataset; print('Dataset loading OK')"

# Test model training
python -c "from sklearn.ensemble import RandomForestClassifier; print('Models OK')"
```

## 📦 Project Structure

```
scientific-paper-classification/
├── 📄 main.py                    # Main pipeline script
├── 📋 requirements.txt           # Python dependencies
├── 📓 Topic_Modeling_SHAP.ipynb  # Extended analysis notebook
├── 📊 label_distribution.png     # Data visualization
├── 📈 word_count_distribution.png # Text analysis charts
├── 📁 cache/                     # Dataset cache directory
└── 📖 README.md                  # Project documentation
```

## 🚀 Performance Benchmarks

### ⚡ Speed Metrics
- **Data Loading**: ~30s (first run), ~5s (cached)
- **Text Preprocessing**: ~10s for 1000 samples
- **BoW/TF-IDF**: ~5s vectorization
- **Embeddings**: ~60s (CPU), ~20s (GPU)
- **Model Training**: ~30s total for all models

### 💾 Memory Usage
- **Dataset**: ~500MB cached
- **Embeddings**: ~200MB for 1000 samples
- **Models**: ~50MB total
- **Peak RAM**: ~2GB during embeddings

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! 

### 📝 Development Workflow
1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

### 🎨 Code Standards
- **Python**: PEP 8 + Black formatter
- **Commits**: Conventional Commits
- **Documentation**: Docstrings cho mọi function
- **Testing**: Unit tests cho critical functions

## 📄 License

MIT License - xem [LICENSE](LICENSE) để biết chi tiết.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) - Datasets và Transformers library
- [Scikit-learn](https://scikit-learn.org/) - Machine learning framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework
- [LightGBM](https://lightgbm.readthedocs.io/) - Microsoft's gradient boosting
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [arXiv](https://arxiv.org/) - Scientific paper dataset

---

<div align="center">

**Được phát triển với ❤️ bởi AI Vietnam Team**

[![GitHub Stars](https://img.shields.io/github/stars/your-repo/scientific-paper-classification?style=social)](https://github.com/your-repo/scientific-paper-classification)
[![GitHub Forks](https://img.shields.io/github/forks/your-repo/scientific-paper-classification?style=social)](https://github.com/your-repo/scientific-paper-classification/fork)

[⭐ Star repo này nếu nó hữu ích!](https://github.com/your-repo/scientific-paper-classification)

</div>