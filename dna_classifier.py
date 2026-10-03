"""
DNA Promotör Sınıflandırma Boru Hattı (Pipeline)
Yöntem: K-mer Sayımı + CountVectorizer + Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def get_kmers(sequence: str, k: int = 4) -> str:
    """
    Bir DNA dizisini k uzunluğundaki alt dizilere (k-mer) böler
    ve aralarına boşluk koyarak NLP kelime dizgisi haline getirir.
    """
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def generate_synthetic_data(n_samples: int = 200, seq_len: int = 100):
    """
    Modeli test etmek için sentetik DNA veri seti üretir.
    Pozitif sınıf (Promotör): TATAAT veya TTGACA benzeri motifler içerir.
    Negatif sınıf (Non-promotör): Tamamen rastgele nükleotidlerden oluşur.
    """
    np.random.seed(42)
    bases = ['A', 'C', 'G', 'T']
    sequences = []
    labels = []

    for i in range(n_samples):
        seq = "".join(np.random.choice(bases, size=seq_len))
        if i % 2 == 0:
            insert_pos = np.random.randint(10, 40)
            seq = seq[:insert_pos] + "TATAAT" + seq[insert_pos+6:]
            labels.append(1)
        else:
            labels.append(0)
        sequences.append(seq)

    return pd.DataFrame({"sequence": sequences, "label": labels})

def run_pipeline():
    print("--- 1. Veri Üretimi ve Hazırlığı ---")
    df = generate_synthetic_data(n_samples=300, seq_len=120)
    print(f"Toplam Veri Sayısı: {len(df)} (Sınıf Dağılımı: {df['label'].value_counts().to_dict()})")

    print("\n--- 2. K-mer Tokenizasyonu (k=4) ---")
    df["kmers"] = df["sequence"].apply(lambda x: get_kmers(x, k=4))

    X_train, X_test, y_train, y_test = train_test_split(
        df["kmers"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    print("\n--- 3. Vektörleştirme (CountVectorizer) ---")
    cv = CountVectorizer(ngram_range=(1, 1))
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)
    print(f"Öznitelik Matrisi Boyutu: {X_train_vec.shape}")

    print("\n--- 4. Random Forest Sınıflandırıcısı Eğitiliyor ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\n--- 5. Performans Sonuçları ---")
    print(f"Doğruluk (Accuracy): {acc:.4f}")
    print(f"ROC-AUC Skoru: {roc:.4f}\n")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=["Non-Promoter (0)", "Promoter (1)"]))

if __name__ == "__main__":
    run_pipeline()