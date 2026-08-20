"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SourceChip from "./SourceChip";
import styles from "./AnswerCard.module.css";

interface Chunk {
  chunk_id: string;
  source: string;
  content_preview: string;
  score: number;
  retrieval_modality: string;
}

interface Props {
  response: {
    answer: string;
    citations: string[];
    sources: string[];
    refusal: string | null;
    confidence: number;
    retrieved_chunks: Chunk[];
    latency_ms: number;
    latency_breakdown: Record<string, number>;
    langfuse_url: string | null;
    prompt_version: string;
  };
}

export default function AnswerCard({ response }: Props) {
  const [chunksExpanded, setChunksExpanded] = useState(false);

  const isRefusal = !!response.refusal;
  const confidencePct = Math.round(response.confidence * 100);

  return (
    <div className={`glass ${styles.card} ${isRefusal ? styles.refusalCard : styles.answerCard}`}>

      {/* ── Meta Bar ── */}
      <div className={styles.metaBar}>
        <div className={styles.metaLeft}>
          <div className={`${styles.statusBadge} ${isRefusal ? styles.statusRefusal : styles.statusSuccess}`}>
            {isRefusal ? "⚠ No Answer" : "✓ Answered"}
          </div>
          {/* A refusal carries no answer confidence — showing one reads as
              "100% sure of nothing", so it is omitted here. */}
          {!isRefusal && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>Confidence</span>
              <span className={styles.metaValue}>{confidencePct}%</span>
            </div>
          )}
          <div className={`${styles.metaItem} ${styles.latencyHoverGroup}`}>
            <span className={styles.metaLabel}>Latency</span>
            <span className={styles.metaValue}>{Math.round(response.latency_ms)}ms</span>
            {response.latency_breakdown && Object.keys(response.latency_breakdown).length > 0 && (
              <div className={styles.latencyTooltip}>
                {Object.entries(response.latency_breakdown).map(([key, val]) => (
                  <div key={key} className={styles.latencyRow}>
                    <span>{key}</span>
                    <span>{val}ms</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Prompt</span>
            <span className={styles.metaValue}>{response.prompt_version}</span>
          </div>
          {response.langfuse_url && (
            <a href={response.langfuse_url} target="_blank" rel="noreferrer" className={styles.traceLink}>
              View Trace ↗
            </a>
          )}
        </div>
        <div className={styles.confidenceBar}>
          <div
            className={styles.confidenceFill}
            style={{ width: `${confidencePct}%`, opacity: isRefusal ? 0.3 : 1 }}
          />
        </div>
      </div>

      {/* ── Answer / Refusal ── */}
      <div className={styles.answerBody}>
        {isRefusal ? (
          <p className={styles.refusalText}>{response.refusal}</p>
        ) : (
          <div className={styles.markdownBody}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {response.answer}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* ── Citations ── */}
      {response.citations.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>📎</span>
            Citations
          </h3>
          <div className={styles.citationList}>
            {response.citations.map((cid) => (
              <code key={cid} className={styles.citationChip}>{cid}</code>
            ))}
          </div>
        </div>
      )}

      {/* ── Sources ── */}
      {response.sources.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>📄</span>
            Sources ({response.sources.length})
          </h3>
          <div className={styles.sourceList}>
            {response.sources.map((s) => (
              <SourceChip key={s} source={s} />
            ))}
          </div>
        </div>
      )}

      {/* ── Retrieved Chunks (Expandable) ── */}
      {response.retrieved_chunks.length > 0 && (
        <div className={styles.section}>
          <button
            className={styles.expandBtn}
            onClick={() => setChunksExpanded(!chunksExpanded)}
            aria-expanded={chunksExpanded}
          >
            <span className={styles.sectionIcon}>🔍</span>
            {chunksExpanded ? "Hide" : "Show"} Retrieved Chunks ({response.retrieved_chunks.length})
            <span className={`${styles.chevron} ${chunksExpanded ? styles.chevronUp : ""}`}>▾</span>
          </button>
          {chunksExpanded && (
            <div className={`${styles.chunksGrid} animate-fade-up`}>
              {response.retrieved_chunks.map((chunk, i) => (
                <div 
                  key={chunk.chunk_id} 
                  className={styles.chunkCard}
                  style={{ animationDelay: `${i * 0.1}s`, animation: `fadeUp 0.4s ease forwards`, opacity: 0 }}
                >
                  <div className={styles.chunkHeader}>
                    <code className={styles.chunkId}>{chunk.chunk_id}</code>
                    <span className={styles.chunkScore}>
                      {(chunk.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span className={styles.modalityBadge}>{chunk.retrieval_modality}</span>
                  <p className={styles.chunkSource}>{chunk.source}</p>
                  <p className={styles.chunkPreview}>{chunk.content_preview}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
