# [2022041046] Jaeyun Byeun - https://github.com/jaeyunbyeun/NLP.git
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


DATA_PATH = "../data/train.csv"  # train.csv 위치


def clean_text(s: str) -> str:
    s = str(s).lower()

    # URL 제거
    s = re.sub(r"http\S+|www\S+", " url ", s)

    # 숫자를 number로 치환
    s = re.sub(r"\d+", " number ", s)

    # 알파벳/숫자/공백 제외는 제거
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 공백 정리
    s = re.sub(r"\s+", " ", s).strip()

    return s


def load_data(path: str):
    df = pd.read_csv(path)
    print("전체 샘플 수:", len(df))

    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    print("\n라벨 평균(비율):")
    print(df[label_cols].mean())

    df["clean"] = df["comment_text"].fillna("").apply(clean_text)
    return df, label_cols


def build_tfidf(train_texts, val_texts):
    tfidf = TfidfVectorizer(
        max_features=120_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.9,
    )

    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)

    return tfidf, X_train, X_val


def train_and_eval(X_train, y_train, X_val, y_val, label_cols):
    preds = {}
    auc_list = []

    for col in label_cols:
        print(f"\n=== {col} 학습 중 ===")

        clf = LogisticRegression(
            max_iter=400,
            n_jobs=-1,
            solver="saga",  # sparse matrix에 적합
        )

        clf.fit(X_train, y_train[col])
        p = clf.predict_proba(X_val)[:, 1]

        preds[col] = p
        auc = roc_auc_score(y_val[col], p)
        auc_list.append(auc)

        print(f"{col} AUC: {auc:.4f}")

    print("\n=== 최종 평균 AUC ===")
    print(f"mean AUC: {np.mean(auc_list):.4f}")


def main():
    df, label_cols = load_data(DATA_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        df["clean"],
        df[label_cols],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"],  # multi-label에서 가장 이상적인 stratify는 아님. 하지만 흔히 쓰는 방식
    )

    tfidf, Xtr, Xv = build_tfidf(X_train, X_val)
    train_and_eval(Xtr, y_train, Xv, y_val, label_cols)


if __name__ == "__main__":
    main()
