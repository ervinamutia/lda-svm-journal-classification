## Penerapan Topic Modeling dengan Latent Dirichlet Allocation (LDA) dan Support Vector Machine (SVM) dalam Klasifikasi Artikel Jurnal

### 📖 Overview
Proyek ini merupakan tugas akhir yang menjadi syarat penyelesaian studi S1 Sistem Informasi.<br>
Tujuan utama penelitian ini menghadirkan sistem klasifikasi untuk artikel jurnal menggunakan pendekatan machine learning untuk mengoptimalkan proses pengelompokan artikel penelitian.

### 🧠 Methods
#### 1. Topic Modeling - LDA
- Ekstraksi topik latent dari teks abstrak
- Menghasilkan distribusi probabilitas topik
- Menentukan topik dominan untuk pelabelan

#### 2. Classification - SVM
- Menggunakan distribusi topik sebagai input fitur
- Mengklasifikasikan artikel ke dalam kategori yang telah ditentukan
- Mengevaluasi kinerja klasifikasi

### ⚙️ Workflow
1. Pengumpulan Data (Scraping)
2. Preprocessing Teks
3. Ekstraksi Fitur menggunakan LDA
4. Pelabelan otomatis berdasarkan topik dominan
6. Klasifikasi data dengan SVM
7. Evaluasi

### 🛠️ Technologies Used
- Python
- Scikit-learn
- Gensim
- Pandas
- NLTK
- Sastrawi
- Sklearn

### 📊 Results
Penerapan kombinasi LDA dan SVM menghasilkan hasil evaluasi berbeda dari kedua tipe data uji, diantaranya:
###### Abstrak saja
  Coherence score: 0.217
  Accuracy: 79% | Precision: 84% | Recall: 79% | F1-score: 81%
###### Abstrak + Keywords
  Coherence score: 0.229
  Accuracy: 76% | Precision: 82% | Recall: 76% | F1-score: 78%
