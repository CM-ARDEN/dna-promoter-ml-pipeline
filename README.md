# Genomik Dizilerde Doğal Dil İşleme Yaklaşımı: K-mer Analizi ile DNA Promotör Bölgesi Sınıflandırması

## 1. Giriş ve Biyolojik Problem

Yeni Nesil Dizileme (Next-Generation Sequencing - NGS) teknolojilerinin gelişmesiyle birlikte biyolojik veri tabanlarındaki genom dizisi hacmi katlanarak artmıştır. Ancak milyonlarca nükleotid diziliminden oluşan bu devasa veri yığını, anotasyon (işaretleme) adımları olmaksızın biyolojik olarak anlam ifade etmez. Genomik anotasyonun en kritik aşamalarından biri, gen ekspresyonunun başlangıç noktasını belirleyen **promotör bölgelerinin** tespit edilmesidir.

Promotörler, RNA polimeraz ve ilgili transkripsiyon faktörlerinin DNA'ya bağlandığı, transkripsiyonun başlatılmasını tetikleyen spesifik düzenleyici dizilerdir (örneğin prokaryotlarda Pribnow kutusu - TATAAT motifi, ökaryotlarda TATA-box). Laboratuvar ortamında bu bölgelerin ıslak laboratuvar (wet-lab) deneyleriyle doğrulanması yüksek maliyetli ve zaman alıcıdır. Bu nedenle, hesaplamalı biyoloji ve makine öğrenmesi (ML) algoritmaları, DNA dizilerinden düzenleyici motifleri ve promotörleri otomatik olarak ayırt etmek için kritik bir rol oynar.

---

## 2. Biyolojik Dizilerin Temsili: DNA'yı NLP ile Modellemek

Geleneksel makine öğrenmesi algoritmaları sayısal matrislerle çalışır. Ancak DNA dizileri `A`, `C`, `G`, `T` harflerinden meydana gelen kategorik metin dizgileridir. Bu dizileri sayısal uzaya aktarmak için Doğal Dil İşleme (NLP) paradigmaları genomik araştırmalara uyarlanır:

1. **K-mer Tokenizasyonu:** Bir DNA dizisi, $k$ uzunluğundaki üst üste binen alt dizilere bölünür. Örneğin `k=4` seçildiğinde `ATGCGT` dizisi `ATGC`, `TGCG`, `GCGT` şeklinde kelimelere ayrılır. Bu yaklaşım, ardışık nükleotid bağlamını ve yerel motif yapısını korur.
2. **Kelime Çantası (Bag of Words - BoW):** Bölünen k-mer dizgileri, `CountVectorizer` yardımıyla bir frekans matrisine dönüştürülür. Her bir k-mer ($4^k$ olası kombinasyon) bir öznitelik (feature) sütununu temsil eder.

Bu temsil şekli, biyolojik dizilerdeki nükleotid komşuluklarını kaybetmeden deterministik bir öznitelik vektörü üretmemizi sağlar.

---

## 3. Boru Hattı Mimarisi (Pipeline Architecture)

Geliştirilen boru hattı uçtan uca şu aşamalardan meydana gelmektedir:

1. **Veri Üretimi / Girişi:** Belirli promotör konsensüs motiflerini içeren pozitif sınıf ile rastgele nükleotid dağılımına sahip negatif dizilerden oluşan veri seti hazırlanır.
2. **Ön İşleme ve K-mer Çıkarımı:** Dizi girdileri kayan pencere (sliding window) mekanizmasıyla $k=4$ boyutundaki k-mer tokenlerine ayrıştırılır.
3. **Eğitim/Test Ayrımı:** Sınıf oranlarını korumak amacıyla veri seti tabakalı (stratified) olarak %75 eğitim ve %25 test olarak bölünür.
4. **Vektörleştirme:** Eğitim verisinden oluşturulan sözlük (vocabulary) üzerinden sparse matris üretilir ve test verisi veri sızıntısını (data leakage) engellemek adına sadece dönüştürülür (`transform`).
5. **Sınıflandırma ve Doğrulama:** Sayısal öznitelikler, doğrusal olmayan ilişkileri ve motif etkileşimlerini yakalamada güçlü olan **Random Forest (Rastgele Orman)** algoritmasına beslenir. Model performansı Accuracy, ROC-AUC ve F1-skoru ile değerlendirilir.

---

## 4. Uygulama ve Çalıştırma

### Gereksinimler
Gerekli kütüphaneleri yüklemek için:
```bash
pip install numpy pandas scikit-learn