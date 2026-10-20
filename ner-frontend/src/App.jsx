
import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [entities, setEntities] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);

  useEffect(() => {
  if (!error) return;

  setIsFadingOut(false);

  const fadeTimer = setTimeout(() => {
    setIsFadingOut(true);
  }, 2000);

  const removeTimer = setTimeout(() => {
    setError("");
    setIsFadingOut(false);
  }, 2500);

  return () => {
    clearTimeout(fadeTimer);
    clearTimeout(removeTimer);
  };
  }, [error]);

  async function handleSubmit(event) {
    event.preventDefault();

    setError("");
    setEntities([]);
    setSubmitted(false);

    if (!title.trim() || !body.trim()) {
      setError("Please enter both the title and article text.");
      return;
    }

    if (`${title}\n${body}`.trim().length > 1000) {
      setError("The article must not exceed 1000 characters.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/extract-entities",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ title, body }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to extract entities.");
      }

      setEntities(data.entities);
      setSubmitted(true);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the API. Make sure FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <p className="eyebrow">NATURAL LANGUAGE PROCESSING</p>
        <h1>Article NER</h1>
        <p className="subtitle">
          Discover people, organizations, and places in your article.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="article-form">
        <label htmlFor="title">Article title</label>
        <input
          id="title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Enter your article title"
          maxLength={1000}
        />

        <label htmlFor="body">Article text</label>
        <textarea
          id="body"
          value={body}
          onChange={(event) => setBody(event.target.value)}
          placeholder="Paste your article here..."
          rows={9}
          maxLength={1000}
        />
        <p className="hint">
        {title.length + body.length} / 1.000 characters
        </p>

        <p className="hint">
          Maximum combined length: 1.000 characters.
        </p>

        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Extract Entities"}
        </button>
      </form>

      {error && (
    <p className={`error ${isFadingOut ? "fade-out" : ""}`}>
      {error}
    </p>
    )}

      {submitted && (
        <section className="results">
          <h2>Extraction Results</h2>
          <p className="result-count">
            Found {entities.length} entities
          </p>

          {entities.length === 0 ? (
            <p>No entities were detected in this article.</p>
          ) : (
            <div className="entity-list">
              {entities.map((entity, index) => (
                <div className="entity" key={`${entity.start}-${index}`}>
                  <div>
                    <strong>{entity.text}</strong>
                    <span className="entity-label">{entity.label}</span>
                  </div>
                  <small>
                    Characters {entity.start}–{entity.end}
                  </small>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

export default App;
