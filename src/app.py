
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

app = FastAPI(title="Article NER API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("models/trained_ner")


class Article(BaseModel):
    title: str
    body: str


@app.get("/")
def home():
    return {"message": "Article NER API is running"}


@app.post("/extract-entities")
def extract_entities(article: Article):
    text = f"{article.title}\n{article.body}".strip()

    if not article.title.strip() or not article.body.strip():
        raise HTTPException(
        status_code=400,
        detail="Title and body cannot be empty"
    )

    if len(text) > 1000:
            raise HTTPException(
            status_code=413,
            detail="Article text must not exceed 1,000 characters"
        )

    
    title_doc = nlp(article.title)
    body_doc = nlp(article.body)

    title_entities = [
    {
        "text": ent.text,
        "label": ent.label_,
        "start": ent.start_char,
        "end": ent.end_char
    }
    for ent in title_doc.ents
]
    
    offset = len(article.title) + 1

    body_entities = [
    {
        "text": ent.text,
        "label": ent.label_,
        "start": ent.start_char + offset,
        "end": ent.end_char + offset
    }
    for ent in body_doc.ents
]

    entities = title_entities + body_entities

    return {
        "entities": entities,
        "entity_count": len(entities),
        "model": "trained_ner"
    }
