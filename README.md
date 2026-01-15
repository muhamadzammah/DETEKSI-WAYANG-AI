Wayang AI Detection System

Wayang AI Detection adalah sistem kecerdasan buatan berbasis Computer Vision dan Deep Learning yang dirancang untuk melakukan klasifikasi tokoh wayang dari citra digital secara otomatis. Sistem ini menggabungkan Convolutional Neural Network (CNN) dengan teknik Transfer Learning MobileNetV2, serta dilengkapi dengan pengolahan citra digital dan antarmuka web interaktif.

Tujuan utama sistem ini adalah membantu pelestarian budaya wayang Indonesia melalui pemanfaatan teknologi AI, sekaligus menjadi media edukasi yang informatif dan mudah digunakan.

🎯 Tujuan Sistem

Sistem ini bertujuan untuk:

Mengklasifikasikan tokoh wayang berdasarkan citra digital

Mengidentifikasi karakter wayang secara cepat dan akurat

Menampilkan deskripsi tokoh wayang sebagai informasi edukatif

Memvisualisasikan tahapan pengolahan citra untuk analisis model

Menjadi implementasi nyata CNN dan Transfer Learning pada budaya lokal

⚙️ Cara Kerja Sistem

Alur kerja sistem secara umum adalah sebagai berikut:

Input Gambar
Pengguna mengunggah citra wayang melalui antarmuka web.

Preprocessing Citra
Sistem melakukan beberapa tahap pengolahan citra:

Resize citra ke ukuran 224×224 piksel

Konversi ke grayscale

Thresholding (Otsu)

Deteksi tepi (Canny, Sobel, Prewitt)

Operasi morfologi (opening dan closing)

Klasifikasi dengan CNN
Citra hasil preprocessing diproses oleh model CNN berbasis MobileNetV2 untuk menentukan kelas tokoh wayang.

Output Hasil Deteksi
Sistem menampilkan:

Nama tokoh wayang

Tingkat keyakinan (confidence score)

Deskripsi tokoh wayang

Visualisasi hasil pengolahan citra

🏗️ Arsitektur Sistem

Sistem terdiri dari tiga komponen utama:

1. Backend (FastAPI)

Mengelola request dan response

Menangani upload gambar

Melakukan preprocessing citra

Menjalankan prediksi model AI

Mengirim hasil dalam format JSON

2. Model AI (TensorFlow & Keras)

Menggunakan MobileNetV2 pretrained ImageNet

Fine-tuning pada layer akhir untuk menyesuaikan karakteristik wayang

Dilengkapi class weighting untuk dataset tidak seimbang

3. Frontend (HTML, Bootstrap, JavaScript)

Antarmuka web responsif dan modern

Upload gambar dan preview

Menampilkan hasil prediksi dan visualisasi citra

📊 Metode yang Digunakan

Convolutional Neural Network (CNN)

Transfer Learning MobileNetV2

Fine-Tuning Model

Data Augmentation

Pengolahan Citra Digital

Class Weighting

Early Stopping & Learning Rate Scheduler

🖼️ Dataset

Dataset terdiri dari citra tokoh wayang dengan berbagai kelas seperti Arjuna, Bima, Semar, Gatotkaca, dan lainnya. Dataset disusun dalam struktur folder berdasarkan nama kelas dan digunakan sebagai input pelatihan model.

Catatan: Dataset tidak disertakan di repository karena keterbatasan ukuran.

🚀 Kelebihan Sistem

Menggunakan model ringan dan efisien (MobileNetV2)

Akurat untuk dataset terbatas

Menyediakan visualisasi pengolahan citra

Antarmuka web interaktif

Mudah dikembangkan dan di-deploy

📚 Potensi Pengembangan

Penambahan jumlah kelas wayang

Integrasi Grad-CAM untuk interpretabilitas model

Deployment ke cloud (Render, Railway, Docker)

Versi mobile atau Progressive Web App (PWA)

🇮🇩 Kontribusi Budaya

Sistem ini diharapkan dapat menjadi sarana digitalisasi dan pelestarian seni wayang sebagai warisan budaya Indonesia melalui penerapan teknologi kecerdasan buatan modern.
