# Article Named Entity Recognition (NER) API & Frontend
Proyek ini merupakan sistem ekstraksi entitas (*Named Entity Recognition*) berbasis AI yang mengekstrak entitas penting dari judul dan isi artikel. Sistem ini dibangun menggunakan **FastAPI** di sisi backend, model **spaCy** yang telah dilatih khusus, serta antarmuka web interaktif berbasis **React + Vite**.

---

## 👥 Pembagian Tugas Tim
* **Anggi**: Pengembangan dan Pelatihan Model NER (`models/trained_ner`).
* **Takeru**: Pengembangan Endpoint API (`POST /extract-entities`), Middleware CORS, dan Sistem Validasi Backend.
* **Vicky**: Implementasi Pengujian Otomatis (*Automated Testing*) & Penyusunan Dokumentasi Proyek (`README.md`).

---

## 📁 Struktur Direktori Proyek

```text
NER-main/
├── data/                             # Dataset dan file evaluasi
│   ├── annotations.jsonl
│   └── evaluation_examples.json
├── models/                           # Model spaCy hasil pelatihan
│   └── trained_ner/
│       ├── config.cfg
│       ├── meta.json
│       ├── tokenizer
│       ├── entity_ruler/
│       ├── ner/
│       └── vocab/
├── ner-frontend/                     # Aplikasi Frontend (React + Vite)
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── src/                             # Kode sumber utama backend / modul
├── test/                            # File Pengujian Otomatis (Automated Tests)
│   └── test_app.py
├── .gitignore
├── app.py                           # Server FastAPI & Endpoint REST API
├── requirements.txt                 # Dependensi Python / Backend
└── README.md                        # Dokumentasi Proyek
```

## ⚙️ Panduan Instalasi
### 1. Prasyarat Sistem
    Python 3.10+
    Node.js 18+ & npm

### 2. Instalasi Backend (FastAPI)
#### 1) Buat dan aktifkan virtual environment:
##### Windows
    python -m venv venv
    venv\Scripts\activate
    
##### Linux / Mac OS
    python3 -m venv venv
    source venv/bin/activate

#### 2) Install seluruh dependensi backend:
    pip install -r requirements.txt

#### 3) Jalankan server backend
    uvicorn app:app --reload

### 3. Instalasi Frontend (React + Vite)
#### 1) Masuk ke folder frontend
    cd ner-frontend

#### 2) Instal depedensi modul node.js
    npm install

#### 3) Jalankan server pengembang frontend
    npm run dev

### 4. Menjalankan Pengujian Otomatis
##### Untuk menjalankan seluruh skenario pengujian otomatis, jalankan perintah berikut di folder utama (root) proyek:
    pytest

## 🚀 Penggunaan API
### 1. Cek Status Server
#### End Point
    GET /
#### Deskripsi 
    Memeriksa apakah API backend berjalan dengan baik.
#### Response (200 OK):
    {
      "message": "Article NER API is running"
    }

### 2. Ekstraksi Entitas
#### End Point
    POST /extract-entities
#### Header
    Content-Type: application/json
#### Request Body
    {
      "title": "Kunjungan Kerja di Jakarta",
      "body": "Joko Widodo mengunjungi Monas pada hari Senin."
    }
#### Response Berhasil (200 OK):
    {
        "entities": [
            {
              "text": "Joko Widodo",
              "label": "PERSON",
              "start": 0,
              "end": 11
            },
        {
              "text": "Monas",
              "label": "LOCATION",
              "start": 24,
              "end": 29
        }
        ],
        "meta": {
        "entity_count": 2,
        "model": "trained_ner"
        }
    }
#### Response Gagal-Input Kosong (400 Bad Request):
    {
      "detail": "Title and body cannot be empty"
    }
#### Response Gagal-Teks Terlalu Panjang (413 Payload Too Large):
    {
      "detail": "Article text must not exceed 1,000 characters"
    }

## ⚠️ Batasan Sistem
### 1. Batas Panjang Karakter: 
    Gabungan teks title dan body tidak boleh melebihi 1.000 karakter. Permintaan yang melebihi batas ini akan ditolak dengan status HTTP 413.
### 2. Validasi Input Kosong: 
    Permintaan dengan field title atau body yang kosong (atau hanya berisi spasi) akan ditolak oleh backend dengan status HTTP 400.

### 3. Aturan CORS: 
    Backend API secara terbatas hanya mengizinkan akses koneksi dari origin lokal frontend (http://localhost:5173 dan http://127.0.0.1:5173).

### 4. Cakupan Model NER: 
    Jenis dan akurasi entitas yang berhasil diekstrak sangat bergantung pada cakupan data latih (annotation dataset) yang terdapat pada direktori models/trained_ner/.
