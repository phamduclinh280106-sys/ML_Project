import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys
import joblib
import os
import re

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



SUSPICIOUS_TLDS = {"ru", "cn", "info", "biz", "tk", "ml", "ga", "cf", "gq"}


SENSITIVE_KEYWORDS = [
    "login", "signin", "secure", "update", "verify",
    "account", "banking", "confirm", "password", "credential",
    "support", "helpdesk", "recover", "unlock", "alert",
    "suspended", "unusual", "activity"
]

BRAND_NAMES = [
    "paypal", "apple", "amazon", "google", "microsoft",
    "facebook", "netflix", "ebay", "bank", "chase"
]



def parse_url(url: str) -> dict:
    """Tách URL thành các phần: domain, path, root, tld."""
    s = str(url).strip().lower()
    s = re.sub(r'https?://', '', s)
    s = re.sub(r'^www\.', '', s)
    parts  = s.split('/', 1)
    domain = parts[0].split('?')[0].split('#')[0].split(':')[0]
    path   = parts[1] if len(parts) > 1 else ''
    dparts = domain.split('.')
    root   = dparts[-2] if len(dparts) >= 2 else dparts[0]
    tld    = dparts[-1] if dparts else ''
    return {"domain": domain, "path": path, "root": root, "tld": tld}


def tokenize_domain(url: str) -> list:
    """
    Tokenizer CHỈ cho phần domain.
    Sensitive keywords được kiểm tra ở đây — an toàn vì không bị ảnh hưởng
    bởi path (github.com/openai, w3.org/tr/wcag, evga.com/articles).
    """
    p      = parse_url(url)
    domain = p["domain"]
    root   = p["root"]
    tld    = p["tld"]
    tokens = []

    full_len = len(str(url).strip())
    if   full_len < 30:  tokens.append("__len_short__")
    elif full_len < 75:  tokens.append("__len_medium__")
    elif full_len < 150: tokens.append("__len_long__")
    else:                tokens.append("__len_verylong__")

    if domain.count('-') >= 3:                   tokens.append("__many_dashes__")
    if domain.count('.')  >= 3:                  tokens.append("__many_dots__")
    if sum(c.isdigit() for c in domain) >= 4:    tokens.append("__many_digits_domain__")
    if "@" in str(url):                          tokens.append("__has_at__")
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}', domain): tokens.append("__ip_address__")

    tokens.append(f"__tld_{tld}__")
    if tld in SUSPICIOUS_TLDS:
        tokens.append("__suspicious_tld__")

    # Sensitive keywords trong domain (không phải path)
    for kw in SENSITIVE_KEYWORDS:
        if kw in domain:
            tokens.append(f"__kw_{kw}__")

    # Brand impersonation: brand có trong domain nhưng root KHÔNG phải brand
    for brand in BRAND_NAMES:
        if brand in domain and root != brand:
            tokens.append(f"__fake_{brand}__")
            tokens.append("__brand_impersonation__")
            break

    # Từ trong domain
    dwords = re.split(r'[^a-zA-Z0-9]', domain)
    tokens.extend([w.lower() for w in dwords if len(w) > 1])

    return tokens


def tokenize_path(url: str) -> list:
    """
    Tokenizer CHỈ cho phần path/query.
    KHÔNG áp dụng sensitive keywords — tránh nhầm path hợp lệ
    như /openai/gpt-4, /tr/wcag10/wai-pageauth, /articles/00438.
    """
    path   = parse_url(url)["path"]
    tokens = []

    if not path:
        tokens.append("__no_path__")
        return tokens

    depth = path.count('/')
    if   depth == 0: tokens.append("__path_depth_0__")
    elif depth <= 2: tokens.append("__path_depth_shallow__")
    elif depth <= 5: tokens.append("__path_depth_medium__")
    else:            tokens.append("__path_depth_deep__")

    if '?' in path:            tokens.append("__has_query__")
    if path.count('=') >= 3:   tokens.append("__many_params__")

    pwords = re.split(r'[^a-zA-Z0-9]', path)
    tokens.extend([w.lower() for w in pwords if len(w) > 1])

    return tokens



class DomainExtractor(BaseEstimator, TransformerMixin):
    """Trích xuất và tokenize phần domain, trả về string cho TfidfVectorizer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join(tokenize_domain(url)) for url in X]


class PathExtractor(BaseEstimator, TransformerMixin):
    """Trích xuất và tokenize phần path, trả về string cho TfidfVectorizer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join(tokenize_path(url)) for url in X]



MODEL_PATH = "models/url_detector_v5.pkl"
pipeline   = None  # được set bởi load_model() hoặc if __name__


def load_model(path: str = MODEL_PATH):
    """Load model — gọi hàm này từ file khác sau khi import Tuna1."""
    global pipeline
    pipeline = joblib.load(path)
    return pipeline
def predict_urls(urls: list) -> list:
        """
        Nhận vào list URL, trả về list dict:
        {"url": ..., "result": "SAFE" | "DANGEROUS", "confidence": float}
        """
        series = pd.Series(urls)
        assert pipeline is not None
        preds  = pipeline.predict(series)
        probas = pipeline.predict_proba(series)

        results = []
        for url, pred, prob in zip(urls, preds, probas):
            results.append({
                "url":        url,
                "result":     "SAFE" if pred == "benign" else "DANGEROUS",
                "confidence": round(max(prob) * 100, 2)
            })
        return results


if __name__ == "__main__":
    os.makedirs("models",  exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Tìm thấy model đã lưu tại: {MODEL_PATH}")
        load_model()
        print("Tải model thành công! Bỏ qua bước huấn luyện.\n")

    else:
        print("Chưa có model. Bắt đầu quy trình huấn luyện...\n")

        root_tk = tk.Tk()
        root_tk.withdraw()
        root_tk.attributes('-topmost', True)

        print("Đang mở cửa sổ chọn file...")
        file_path = filedialog.askopenfilename(
            title="Chọn tập dữ liệu URL",
            filetypes=[("Excel/CSV Files", "*.xlsx *.xls *.csv"), ("All Files", "*.*")]
        )

        if not file_path:
            print("Bạn chưa chọn file.")
            sys.exit()

        
        try:
            df = pd.read_csv(file_path, encoding="latin1") if file_path.endswith(".csv") \
                 else pd.read_excel(file_path)
            df.columns = df.columns.str.lower()
            if "type" in df.columns:
                df.rename(columns={"type": "label"}, inplace=True)
        except Exception as e:
            print(f"Lỗi đọc file: {e}"); sys.exit()

        if "url" not in df.columns or "label" not in df.columns:
            print("Thiếu cột 'url' hoặc 'label'"); sys.exit()

        df.dropna(inplace=True)
        df["url"] = df["url"].str.strip().str.lower()

        before = len(df)
        df.drop_duplicates(subset=["url"], inplace=True)
        print(f"Loại bỏ {before - len(df):,} URL trùng lặp")
        df = df[df["url"].str.len() >= 7]

        print(f"Đã nạp {len(df):,} dòng dữ liệu sạch")
        print("\nPhân phối nhãn:")
        print(df["label"].value_counts())

        X = df["url"]
        y = df["label"]

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

        
        domain_tfidf = Pipeline([
            ("extract", DomainExtractor()),
            ("tfidf",   TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\S+",
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
                max_features=10000,
                ngram_range=(1, 2)
            ))
        ])

        path_tfidf = Pipeline([
            ("extract", PathExtractor()),
            ("tfidf",   TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\S+",
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
                max_features=5000,
                ngram_range=(1, 2)
            ))
        ])

        features = FeatureUnion([
            ("domain", domain_tfidf),
            ("path",   path_tfidf)
        ])

       
        log_reg = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1.0,
            random_state=42,
            class_weight="balanced"
        )

        lgbm = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            learning_rate=0.05,
            n_estimators=1000,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced"
        )

        ensemble = VotingClassifier(
            estimators=[("lr", log_reg), ("lgbm", lgbm)], # type: ignore
            voting="soft",
            weights=[1, 2]
        )

    
        pipeline = Pipeline([
            ("features", features),
            ("ensemble", ensemble)
        ])


        print("\nĐang chạy cross-validation trên mẫu 20%...")
        _, X_cv, _, y_cv = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
        )
        cv_scores = cross_val_score(
            pipeline, X_cv, y_cv,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_weighted",
            n_jobs=1,
            error_score="raise"
        )
        print(f"CV F1 (weighted, 3-fold, 20% sample): "
              f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


        pipeline.fit(X_train, y_train)
        print("Huấn luyện hoàn tất!")

        joblib.dump(pipeline, MODEL_PATH, compress=3)
        print(f"\n Đã lưu model tại: {MODEL_PATH}")
    


        print("Đang đánh giá trên tập test...")
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        classes = pipeline.classes_
        auc = roc_auc_score(
            y_test, y_proba,
            multi_class="ovr", average="weighted"
        ) if len(classes) > 2 else roc_auc_score(y_test, y_proba[:, 1])
        print(f"AUC-ROC (weighted OvR): {auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix — URL Detector v5")
        plt.tight_layout()
        plt.savefig("reports/confusion_matrix_v5.png", dpi=120)
        plt.close()
        print("Confusion matrix đã lưu: reports/confusion_matrix_v5.png")

        # Phân tích false positive
        wrong_mask = y_pred != y_test.values
        wrong_df = pd.DataFrame({
            "url":       X_test.values[wrong_mask],
            "actual":    y_test.values[wrong_mask],
            "predicted": y_pred[wrong_mask]
        })
        fp = wrong_df[(wrong_df["actual"] == "benign") & (wrong_df["predicted"] == "phishing")]
        print(f"\nFalse positive phishing (benign → phishing): {len(fp):,}")

        wrong_df.sample(min(500, len(wrong_df)), random_state=42).to_csv(
            "reports/misclassified_v5.csv", index=False
        )
        print("reports/misclassified_v5.csv")


    test_urls = [
    "http://secure-login-appleid-update.com",
    "https://www.google.com/search?q=machine+learning",
    "https://facebook.com",
    "http://paypal-security-update-login.net",
    "http://192.168.1.1/admin",
    "https://github.com/openai/gpt-4",
    "http://amaz0n-account-verify.ru/login",
    "https://evga.com/articles/00438/",
    "https://w3.org/tr/wcag10/wai-pageauth.html",
    ]

    print("\n===== KẾT QUẢ KIỂM TRA URL =====\n")
    for r in predict_urls(test_urls):
        flag = "OK" if r["result"] == "SAFE" else "NO"
        print(f"{flag} {r['result']:<10} ({r['confidence']:5.1f}%)  {r['url']}")

    print(f"\n Muốn train lại: del {MODEL_PATH}  (Windows) / rm {MODEL_PATH}  (Linux/Mac)")
