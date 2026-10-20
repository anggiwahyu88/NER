
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

app = FastAPI(title="Article NER API")

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

    doc = nlp(text)

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
    ]

    return {
        "entities": entities,
        "entity_count": len(entities),
        "model": "trained_ner"
    }
