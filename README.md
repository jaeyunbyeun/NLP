# 💬 악성 댓글 분류를 위한 TF-IDF 기반 Logistic Regression 분석

## 🚀 프로젝트 개요 (Overview)

본 프로젝트는 Kaggle Toxic Comment Classification Challenge 데이터셋을 활용하여 악성 댓글(Toxic Comments)을 자동으로 분류하고 탐지하는 경량 모델을 구축하고 그 성능을 분석합니다.

복잡하고 자원 소모가 큰 최신 Transformer 기반 모델 대신, **TF-IDF(Term Frequency–Inverse Document Frequency)** 벡터화와 **Logistic Regression**을 결합한 전통적인 머신러닝 접근 방식을 채택하여 **높은 효율성**과 **뛰어난 해석 가능성**을 입증하는 것을 목표로 합니다.

## 🌟 주요 특징 및 결과 (Features & Results)

* **모델 아키텍처:** 6개의 독립적인 라벨(toxic, severe toxic, obscene, threat, insult, identity hate)에 대한 Multi-label Logistic Regression 모델 (Binary Relevance).
* **성능 지표:** ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)
* **주요 결과:** 모든 라벨에 대한 평균 **ROC-AUC 0.9795** 달성.
* **최적화:** 소수 클래스 불균형 완화를 위한 `class_weight` 자동 보정 옵션 적용.

| Label | AUC Score |
| :--- | :--- |
| **Average** | **0.9795** |
| severe toxic | 0.9868 |
| toxic | 0.9717 |

## 🛠️ 기술 스택 (Tech Stack)

* **언어:** Python
* **핵심 라이브러리:** `scikit-learn` (LogisticRegression, TfidfVectorizer)
* **데이터 처리:** `pandas`, `numpy`
* **환경:** Jupyter Notebook / Google Colab

## 📁 프로젝트 디렉토리 구조 (Project Directory Structure)

project/
 ├─ data/
 │    ├─ train.csv             
 ├─ src/
 │    ├─ baseline.py           
 │    ├─ bert_train.py        
 ├─ README.md                
 └─ requirements.txt         
