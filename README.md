# NLP Toxic Comment Classification  
Logistic Regression + TF-IDF 기반 다중 라벨 독성 댓글 분류 모델

이 프로젝트는 Kaggle **Toxic Comment Classification Challenge** 데이터셋을 활용하여  
TF-IDF 특징 추출과 **Logistic Regression**을 기반으로 한 다중 라벨 분류 모델을 구현한 코드입니다.

---

## 📌 주요 기능

- **텍스트 전처리(clean_text)**  
  - URL 제거  
  - 숫자 치환(number)  
  - 특수문자 제거  
  - 공백 정리  

- **TF-IDF 벡터화**  
  - 최대 120,000개의 feature  
  - bigram(1–2-gram) 적용  
  - sublinear TF 활성화  

- **Logistic Regression 모델 학습**  
  - 각 라벨별 개별 모델 학습  
  - `solver="saga"`로 sparse matrix 최적화  
  - ROC-AUC 기반 성능 평가  

- **다중 라벨 toxic classification** 수행  
  - toxic  
  - severe_toxic  
  - obscene  
  - threat  
  - insult  
  - identity_hate  

---
