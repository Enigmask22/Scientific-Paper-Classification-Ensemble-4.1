# ==============================================================================
# AI VIET NAM – AI COURSE 2025
# Project: Phân loại chủ đề bài báo dựa vào kỹ thuật Ensemble Learning
# Source Code
# ==============================================================================

# ------------------------------------------------------------------------------
# II.1. Cài đặt và Import thư viện
# ------------------------------------------------------------------------------
print("Step 1: Importing libraries...")

import re
import warnings
from collections import Counter, defaultdict
from typing import List, Dict, Literal, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for data and models
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
CACHE_DIR = "./cache"

# ------------------------------------------------------------------------------
# II.2. Đọc và khám phá bộ dữ liệu
# ------------------------------------------------------------------------------
print("\nStep 2: Loading and exploring the dataset...")
# Tải bộ dữ liệu từ Hugging Face
ds = load_dataset("UniverseTBD/arxiv-abstracts-large", cache_dir=CACHE_DIR)

# Lấy 1000 samples với nhãn duy nhất thuộc các categories được chọn
print("\nStep 3: Sampling data...")
samples = []
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

# Vòng lặp để thu thập các mẫu hợp lệ
for s in ds['train']:
    # Chỉ chọn các bài báo có một category duy nhất
    if len(s['categories'].split()) == 1:
        cur_category = s['categories'].strip().split('.')[0]
        if cur_category in CATEGORIES_TO_SELECT:
            samples.append(s)
    
    # Dừng lại khi đã thu thập đủ 1000 mẫu
    if len(samples) >= 1000:
        break

print(f"Number of samples collected: {len(samples)}")

# ------------------------------------------------------------------------------
# II.3. Tiền xử lý dữ liệu
# ------------------------------------------------------------------------------
print("\nStep 4: Preprocessing data...")
preprocessed_samples = []
for s in samples:
    # Lấy nội dung abstract
    abstract = s['abstract']
    
    # Loại bỏ ký tự \n và khoảng trắng thừa ở đầu/cuối
    abstract = abstract.strip().replace("\n", " ")
    
    # Loại bỏ các ký tự đặc biệt (không phải chữ, số, hoặc khoảng trắng)
    abstract = re.sub(r'[^\w\s]', '', abstract)
    
    # Loại bỏ chữ số
    abstract = re.sub(r'\d+', '', abstract)
    
    # Loại bỏ khoảng trắng thừa
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    
    # Chuyển thành chữ thường
    abstract = abstract.lower()
    
    # Lấy nhãn chính (primary category)
    parts = s['categories'].split(' ')
    category = parts[0].split('.')[0]
    
    preprocessed_samples.append({
        "text": abstract,
        "label": category
    })

# Tạo ánh xạ từ nhãn (string) sang ID (integer)
all_labels = [sample['label'] for sample in preprocessed_samples]
sorted_labels = sorted(list(set(all_labels)))

label_to_id = {label: i for i, label in enumerate(sorted_labels)}
id_to_label = {i: label for i, label in enumerate(sorted_labels)}

print("\nLabel to ID mapping:")
for label, id_ in label_to_id.items():
    print(f"{label} --> {id_}")

# Chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test)
X_full = [sample['text'] for sample in preprocessed_samples]
y_full = [label_to_id[sample['label']] for sample in preprocessed_samples]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ------------------------------------------------------------------------------
# II.4. Mã hóa văn bản (Text Vectorization)
# ------------------------------------------------------------------------------
print("\nStep 5: Vectorizing text data...")

# II.4.1. Bag-of-Words (BoW)
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# II.4.2. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# II.4.3. Sentence Embeddings
class EmbeddingVectorizer:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base', normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def _format_inputs(self, texts: List[str], mode: Literal['query', 'passage']) -> List[str]:
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]

    def transform(self, texts: List[str], mode: Literal['query', 'passage', 'raw'] = 'query') -> np.ndarray:
        if mode == 'raw':
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)
        
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return np.array(embeddings)

embedding_vectorizer = EmbeddingVectorizer()
X_train_embeddings = embedding_vectorizer.transform(X_train)
X_test_embeddings = embedding_vectorizer.transform(X_test)

# Chuyển đổi tất cả sang numpy array để nhất quán
X_train_bow = X_train_bow.toarray()
X_test_bow = X_test_bow.toarray()
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# In ra kích thước của các tập dữ liệu đã được mã hóa
print(f"Shape of X_train_bow: {X_train_bow.shape}")
print(f"Shape of X_test_bow: {X_test_bow.shape}\n")
print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}\n")
print(f"Shape of X_train_embeddings: {X_train_embeddings.shape}")
print(f"Shape of X_test_embeddings: {X_test_embeddings.shape}\n")


# ------------------------------------------------------------------------------
# II.5. Huấn luyện và đánh giá mô hình
# ------------------------------------------------------------------------------
print("\nStep 6: Training and evaluating models...")

# II.5.1. Random Forest
def train_and_test_random_forest(X_train, y_train, X_test, y_test, n_estimators: int = 100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)
    return y_pred, accuracy, report

_, rf_bow_accuracy, _ = train_and_test_random_forest(X_train_bow, y_train, X_test_bow, y_test)
_, rf_tfidf_accuracy, _ = train_and_test_random_forest(X_train_tfidf, y_train, X_test_tfidf, y_test)
_, rf_embeddings_accuracy, _ = train_and_test_random_forest(X_train_embeddings, y_train, X_test_embeddings, y_test)

print("--- Accuracies for Random Forest ---")
print(f"Bag of Words: {rf_bow_accuracy:.4f}")
print(f"TF-IDF: {rf_tfidf_accuracy:.4f}")
print(f"Embeddings: {rf_embeddings_accuracy:.4f}\n")

# II.5.2. AdaBoost
def train_and_test_adaboost(X_train, y_train, X_test, y_test, n_estimators: int = 50):
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)
    return y_pred, accuracy, report

_, ada_bow_accuracy, _ = train_and_test_adaboost(X_train_bow, y_train, X_test_bow, y_test)
_, ada_tfidf_accuracy, _ = train_and_test_adaboost(X_train_tfidf, y_train, X_test_tfidf, y_test)
_, ada_embeddings_accuracy, _ = train_and_test_adaboost(X_train_embeddings, y_train, X_test_embeddings, y_test)

print("--- Accuracies for AdaBoost ---")
print(f"Bag of Words: {ada_bow_accuracy:.4f}")
print(f"TF-IDF: {ada_tfidf_accuracy:.4f}")
print(f"Embeddings: {ada_embeddings_accuracy:.4f}\n")

# II.5.3. Gradient Boosting
def train_and_test_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators: int = 100):
    gb = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)
    return y_pred, accuracy, report

_, gb_bow_accuracy, _ = train_and_test_gradient_boosting(X_train_bow, y_train, X_test_bow, y_test)
_, gb_tfidf_accuracy, _ = train_and_test_gradient_boosting(X_train_tfidf, y_train, X_test_tfidf, y_test)
_, gb_embeddings_accuracy, _ = train_and_test_gradient_boosting(X_train_embeddings, y_train, X_test_embeddings, y_test)

print("--- Accuracies for Gradient Boosting ---")
print(f"Bag of Words: {gb_bow_accuracy:.4f}")
print(f"TF-IDF: {gb_tfidf_accuracy:.4f}")
print(f"Embeddings: {gb_embeddings_accuracy:.4f}\n")

# II.5.4. XGBoost
def train_and_test_xgboost(X_train, y_train, X_test, y_test, n_estimators: int = 100):
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)
    return y_pred, accuracy, report

_, xgb_bow_accuracy, _ = train_and_test_xgboost(X_train_bow, y_train, X_test_bow, y_test)
_, xgb_tfidf_accuracy, _ = train_and_test_xgboost(X_train_tfidf, y_train, X_test_tfidf, y_test)
_, xgb_embeddings_accuracy, _ = train_and_test_xgboost(X_train_embeddings, y_train, X_test_embeddings, y_test)

print("--- Accuracies for XGBoost ---")
print(f"Bag of Words: {xgb_bow_accuracy:.4f}")
print(f"TF-IDF: {xgb_tfidf_accuracy:.4f}")
print(f"Embeddings: {xgb_embeddings_accuracy:.4f}\n")

# II.5.5. LightGBM
def train_and_test_lightgbm(X_train, y_train, X_test, y_test, n_estimators: int = 100):
    lgbm = lgb.LGBMClassifier(
        boosting_type='gbdt', # 'goss' is also an option as mentioned in some documents, but gbdt is default and robust
        n_estimators=n_estimators, 
        random_state=42
    )
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=sorted_labels, output_dict=True)
    return y_pred, accuracy, report

_, lgbm_bow_accuracy, _ = train_and_test_lightgbm(X_train_bow, y_train, X_test_bow, y_test)
_, lgbm_tfidf_accuracy, _ = train_and_test_lightgbm(X_train_tfidf, y_train, X_test_tfidf, y_test)
_, lgbm_embeddings_accuracy, _ = train_and_test_lightgbm(X_train_embeddings, y_train, X_test_embeddings, y_test)

print("--- Accuracies for LightGBM ---")
print(f"Bag of Words: {lgbm_bow_accuracy:.4f}")
print(f"TF-IDF: {lgbm_tfidf_accuracy:.4f}")
print(f"Embeddings: {lgbm_embeddings_accuracy:.4f}\n")

print("Project execution completed.")