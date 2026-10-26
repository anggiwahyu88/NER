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
    1) Buat dan aktifkan virtual environment:
  #### Windows
    python -m venv venv
    venv\Scripts\activate
  #### Linux/Mac OS
    python3 -m venv venv
    source venv/bin/activate

  ### 2) Install seluruh dependensi backend:
  #### Bash
    pip install -r requirements.txt

  4) Jalankan server backend
  #### Bash
     uvicorn app:app --reload

### 3. Instalasi Frontend (React + Vite)
   
