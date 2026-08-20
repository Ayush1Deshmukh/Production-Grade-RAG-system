"use client";

import { useState, useRef, KeyboardEvent } from "react";
import styles from "./QueryInput.module.css";

interface Props {
  onSubmit: (question: string) => void;
  loading: boolean;
}

const EXAMPLE_QUERIES = [
  "What embedding model does this RAG system use?",
  "What is the dimensionality of the dense vector embeddings?",
  "How are the vector embeddings generated?",
  "Which sentence-transformer is used for the RAG architecture?",
];

export default function QueryInput({ onSubmit, loading }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExampleClick = (q: string) => {
    setValue(q);
    textareaRef.current?.focus();
  };

  return (
    <div className={styles.wrapper}>
      {/* Explicit label: an unlabelled textarea reads as decoration, not as
          the place you type your question. */}
      <div className={styles.searchHeading}>
        <h2 className={styles.searchTitle}>Ask a question</h2>
        <p className={styles.searchHint}>
          Searches the indexed documents and answers with citations.
        </p>
      </div>

      <div className={`glass ${styles.inputPanel} ${loading ? styles.loading : ""}`}>
        <div className={styles.inputRow}>
          <span className={styles.prompt} aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <circle cx="11" cy="11" r="7" />
              <path d="M20 20l-3.6-3.6" strokeLinecap="round" />
            </svg>
          </span>
          <textarea
            ref={textareaRef}
            id="rag-query-input"
            className={styles.textarea}
            placeholder="Ask anything about the indexed knowledge base..."
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            disabled={loading}
            aria-label="Query input"
          />
          <button
            id="rag-submit-btn"
            className={`${styles.submitBtn} ${loading ? styles.submitLoading : ""}`}
            onClick={handleSubmit}
            disabled={!value.trim() || loading}
            aria-label="Search"
          >
            {loading ? (
              <span className={styles.spinner} />
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" />
              </svg>
            )}
            <span className={styles.submitLabel}>{loading ? "Searching…" : "Search"}</span>
          </button>
        </div>

        <div className={styles.footer}>
          <span className={styles.hint}>Press Enter to send · Shift+Enter for newline</span>
          <span className={styles.charCount}>{value.length}/2000</span>
        </div>
      </div>

      {/* Example queries */}
      <div className={styles.examples}>
        <span className={styles.examplesLabel}>Try:</span>
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            className={styles.exampleChip}
            onClick={() => handleExampleClick(q)}
            disabled={loading}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
