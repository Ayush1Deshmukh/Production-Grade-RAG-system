"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import styles from "./ArchitectureDiagram.module.css";

interface Chunk {
  chunk_id: string;
  source: string;
  content_preview: string;
  score: number;
  retrieval_modality: string;
}

export interface PipelineData {
  latency_breakdown: Record<string, number>;
  retrieved_chunks: Chunk[];
  citations: string[];
  refusal: string | null;
  prompt_version: string;
}

interface Props {
  loading: boolean;
  data: PipelineData | null;
}

/** A measured connector between two blocks. */
interface Wire {
  d: string;
  from: NodeId;
  to: NodeId;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type NodeId =
  | "client"
  | "api"
  | "embed"
  | "dense"
  | "sparse"
  | "fusion"
  | "dedupe"
  | "rerank"
  | "llm"
  | "answer";

interface DiagramNode {
  id: NodeId;
  band: number;
  label: string;
  tech: string;
  /** Monospace line shown inside the node. */
  code: string;
  /** Expanded snippet revealed on hover, typed in. */
  snippet: string;
  detail: string;
  timingKey?: string;
  accent?: "dense" | "sparse" | "llm";
}

const BANDS = ["Client", "API", "Retrieval", "Ranking", "Generation"];

const NODES: DiagramNode[] = [
  {
    id: "client",
    band: 0,
    label: "Next.js UI",
    tech: "App Router · CSS Modules",
    code: "POST /api/v1/query",
    snippet: 'fetch(`${API}/api/v1/query`, {\n  body: JSON.stringify({ question })\n})',
    detail: "The browser posts the raw question. No retrieval logic lives client-side.",
  },
  {
    id: "api",
    band: 1,
    label: "FastAPI",
    tech: "Pydantic validation",
    code: "QueryRequest(min_length=3)",
    snippet: "class QueryRequest(BaseModel):\n    question: str = Field(min_length=3,\n                          max_length=2000)",
    detail:
      "Requests are schema-validated before any model runs — a malformed question is rejected with 422, never embedded.",
  },
  {
    id: "embed",
    band: 2,
    label: "Embedder",
    tech: "all-MiniLM-L6-v2 · local",
    code: "embed_query(q) -> float[384]",
    snippet: "await run_in_threadpool(\n    get_embeddings().embed_query, question\n)",
    detail:
      "384-dimensional dense vector, computed on-device. It runs in a worker thread so the event loop stays free.",
    timingKey: "Embedding",
  },
  {
    id: "dense",
    band: 2,
    label: "Qdrant",
    tech: "Dense · cosine",
    code: "search(vector, k=10)",
    snippet: "vector_store.as_retriever(\n    search_kwargs={'k': settings.top_k}\n)",
    detail: "Semantic neighbours from the vector index — catches paraphrase, misses exact keywords.",
    accent: "dense",
  },
  {
    id: "sparse",
    band: 2,
    label: "BM25",
    tech: "rank-bm25 · in-process",
    code: "rebuild_bm25_from_collection()",
    snippet: "BM25Retriever.from_documents(\n    _bm25_documents  # rebuilt at startup\n)",
    detail:
      "Sparse keyword scoring over the same chunks, rebuilt from Qdrant on boot so hybrid survives a restart.",
    accent: "sparse",
  },
  {
    id: "fusion",
    band: 2,
    label: "RRF Fusion",
    tech: "EnsembleRetriever · 0.5 / 0.5",
    code: "score = 1 / (k + rank), k=60",
    snippet: "EnsembleRetriever(\n    retrievers=[dense, bm25],\n    weights=[0.5, 0.5],\n)",
    detail:
      "Reciprocal Rank Fusion merges both result lists by rank, not by raw score — so incomparable scales never fight.",
    timingKey: "Hybrid Search",
  },
  {
    id: "dedupe",
    band: 3,
    label: "Deduplicate",
    tech: "content hashing",
    code: "seen.add(text.strip()[:100])",
    snippet: "if content_hash not in seen_content:\n    seen_content.add(content_hash)\n    base_docs.append(doc)",
    detail: "Scraped nav bars and repeated boilerplate are dropped before they can eat the context window.",
  },
  {
    id: "rerank",
    band: 3,
    label: "Cross-Encoder",
    tech: "ms-marco-MiniLM-L-6-v2",
    code: "score(pairs) -> logits",
    snippet: "pairs = [[question, d.page_content] for d in docs]\nscores = await run_in_threadpool(\n    cross_encoder.score, pairs)",
    detail:
      "Question and chunk are scored jointly — far sharper than comparing two independent embeddings. Top-N survive.",
    timingKey: "Reranking",
  },
  {
    id: "llm",
    band: 4,
    label: "Gemini",
    tech: "gemini-3.5-flash-lite · json_mode",
    code: "with_structured_output(RAGResponse)",
    snippet: "structured_llm = llm.with_structured_output(\n    RAGResponse, method='json_mode')",
    detail:
      "The model is bound to a Pydantic schema, so it must emit answer + citations + refusal. Transient 429s are retried with backoff.",
    timingKey: "LLM Generation",
    accent: "llm",
  },
  {
    id: "answer",
    band: 4,
    label: "Cited Answer",
    tech: "grounded or refused",
    code: "citations: [CHUNK_ID, ...]",
    snippet: "{'answer': ..., 'citations': ['chunk_a1b2'],\n 'refusal': None, 'confidence': 0.94}",
    detail:
      "Every claim carries a chunk id. When the context cannot support an answer the model refuses instead of inventing one.",
  },
];

/** Directed edges, drawn as measured SVG wires. */
const EDGES: [NodeId, NodeId][] = [
  ["client", "api"],
  ["api", "embed"],
  ["embed", "dense"],
  ["embed", "sparse"],
  ["dense", "fusion"],
  ["sparse", "fusion"],
  ["fusion", "dedupe"],
  ["dedupe", "rerank"],
  ["rerank", "llm"],
  ["llm", "answer"],
];

/** Order used for the "current stage" sweep while a query is in flight. */
const SWEEP: NodeId[] = [
  "client", "api", "embed", "dense", "sparse", "fusion", "dedupe", "rerank", "llm", "answer",
];

const FLOATING_CODE = [
  "RetrievalMode.HYBRID",
  "chunk_overlap=100",
  "top_k=10",
  "cosine(q, d)",
  "rerank_top_n=5",
  "faithfulness >= 0.90",
  "X-API-Key",
  "384-d",
];

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const on = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

export default function ArchitectureDiagram({ loading, data }: Props) {
  const boardRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Partial<Record<NodeId, HTMLButtonElement | null>>>({});
  const reduced = useReducedMotion();

  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<NodeId | null>(null);
  const [sweepIndex, setSweepIndex] = useState(-1);
  const [wires, setWires] = useState<Wire[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  /* ── Measure node positions and build the connector paths ─────────────── */
  const measure = useCallback(() => {
    const board = boardRef.current;
    if (!board) return;
    const b = board.getBoundingClientRect();
    setSize({ w: b.width, h: b.height });

    const center = (id: NodeId) => {
      const el = nodeRefs.current[id];
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {
        top: { x: r.left - b.left + r.width / 2, y: r.top - b.top },
        bottom: { x: r.left - b.left + r.width / 2, y: r.bottom - b.top },
      };
    };

    const paths: Wire[] = [];
    for (const [from, to] of EDGES) {
      const a = center(from);
      const z = center(to);
      if (!a || !z) continue;
      const start = a.bottom;
      const end = z.top;
      // Vertical cubic with horizontal easing, so forks read as branches.
      const dy = Math.max(18, (end.y - start.y) / 2);
      paths.push({
        from,
        to,
        x1: start.x,
        y1: start.y,
        x2: end.x,
        y2: end.y,
        d: `M ${start.x} ${start.y} C ${start.x} ${start.y + dy}, ${end.x} ${end.y - dy}, ${end.x} ${end.y}`,
      });
    }
    setWires(paths);
  }, []);

  useLayoutEffect(() => {
    measure();
    const ro = new ResizeObserver(measure);
    if (boardRef.current) ro.observe(boardRef.current);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [measure, data]);

  /* ── Stage sweep while loading ────────────────────────────────────────── */
  useEffect(() => {
    if (!loading) {
      setSweepIndex(-1);
      return;
    }
    setSweepIndex(0);
    const id = setInterval(() => setSweepIndex((i) => (i + 1) % SWEEP.length), 380);
    return () => clearInterval(id);
  }, [loading]);

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (reduced || !boardRef.current) return;
    const r = boardRef.current.getBoundingClientRect();
    setTilt({
      x: -((e.clientY - r.top) / r.height - 0.5) * 9,
      y: ((e.clientX - r.left) / r.width - 0.5) * 12,
    });
  };

  const timings = data?.latency_breakdown ?? {};
  const totalTimed = Object.values(timings).reduce((a, b) => a + b, 0);

  const metricFor = (node: DiagramNode): string | null => {
    if (!data) return null;
    switch (node.id) {
      case "api":
        return `prompt ${data.prompt_version}`;
      case "fusion":
        return data.retrieved_chunks[0]?.retrieval_modality.includes("Hybrid") ? "hybrid" : "dense only";
      case "dedupe":
      case "rerank":
        return `${data.retrieved_chunks.length} chunks`;
      case "answer":
        return data.refusal ? "refused" : `${data.citations.length} cited`;
      default:
        return null;
    }
  };

  const activeId = loading && sweepIndex >= 0 ? SWEEP[sweepIndex] : null;

  const renderNode = (node: DiagramNode) => {
    const ms = node.timingKey ? timings[node.timingKey] : undefined;
    const isActive = activeId === node.id;
    const isDone = !loading && !!data;
    const isHovered = hovered === node.id;

    return (
      <div className={styles.nodeWrap} key={node.id}>
        <button
          type="button"
          ref={(el) => { nodeRefs.current[node.id] = el; }}
          className={[
            styles.node,
            node.accent ? styles[node.accent] : "",
            isActive ? styles.active : "",
            isDone ? styles.done : "",
          ].join(" ")}
          onMouseEnter={() => setHovered(node.id)}
          onMouseLeave={() => setHovered(null)}
          onFocus={() => setHovered(node.id)}
          onBlur={() => setHovered(null)}
          aria-describedby={`arch-detail-${node.id}`}
        >
          <span className={styles.nodeHead}>
            <span className={styles.nodeLabel}>{node.label}</span>
            {ms !== undefined && <span className={styles.timing}>{Math.round(ms)}ms</span>}
            {metricFor(node) && <span className={styles.metric}>{metricFor(node)}</span>}
          </span>
          <span className={styles.nodeTech}>{node.tech}</span>
          <span className={styles.nodeCode}>
            {node.code}
            <i className={styles.caret} aria-hidden="true" />
          </span>
          <span className={styles.slab} aria-hidden="true" />
        </button>

        <div
          id={`arch-detail-${node.id}`}
          role="tooltip"
          className={`${styles.panel} ${isHovered ? styles.panelOpen : ""}`}
        >
          <pre className={styles.snippet} key={isHovered ? "open" : "closed"}>
            {node.snippet}
          </pre>
          <p className={styles.detailText}>{node.detail}</p>
        </div>
      </div>
    );
  };

  return (
    <section className={styles.wrapper} aria-label="System architecture">
      <header className={styles.heading}>
        <h2 className={styles.title}>System Architecture</h2>
        <p className={styles.subtitle}>
          {data
            ? `Live trace · ${Math.round(totalTimed)}ms across timed stages`
            : "Hover any component for its code path. Retrieval forks into dense and sparse, then fuses."}
        </p>
        <div className={styles.legend}>
          <span className={styles.legendItem}>
            <i className={`${styles.swatch} ${styles.swatchDense}`} /> dense vector
          </span>
          <span className={styles.legendItem}>
            <i className={`${styles.swatch} ${styles.swatchSparse}`} /> sparse BM25
          </span>
          <span className={styles.legendItem}>
            <i className={`${styles.swatch} ${styles.swatchFlow}`} /> data flow
          </span>
        </div>
      </header>

      <div
        ref={boardRef}
        className={styles.board}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setTilt({ x: 0, y: 0 })}
        style={{ "--tx": `${tilt.x}deg`, "--ty": `${tilt.y}deg` } as React.CSSProperties}
      >
        {/* Drifting code fragments, purely atmospheric */}
        {!reduced && (
          <div className={styles.codeField} aria-hidden="true">
            {FLOATING_CODE.map((t, i) => (
              <span key={t} className={styles.floatCode} style={{ "--i": i } as React.CSSProperties}>
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Measured connector wires */}
        <svg
          className={styles.wires}
          width={size.w}
          height={size.h}
          viewBox={`0 0 ${size.w || 1} ${size.h || 1}`}
          aria-hidden="true"
        >
          <defs>
            <linearGradient id="wireGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(139,92,246,0.9)" />
              <stop offset="100%" stopColor="rgba(6,182,212,0.9)" />
            </linearGradient>
            <marker
              id="arrowHead"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 9 5 L 0 9 z" fill="rgba(139,92,246,0.75)" />
            </marker>
          </defs>
          {wires.map((w) => {
            const lit = activeId === w.from || activeId === w.to;
            return (
              <g key={`${w.from}-${w.to}`}>
                <path className={styles.wire} d={w.d} markerEnd="url(#arrowHead)" />
                <path
                  className={`${styles.wireFlow} ${lit ? styles.wireLit : ""} ${
                    loading && !reduced ? styles.wireRunning : ""
                  }`}
                  d={w.d}
                />
                {/* Ports where the wire meets each block */}
                <circle className={styles.port} cx={w.x1} cy={w.y1} r="2.6" />
                <circle
                  className={`${styles.port} ${lit ? styles.portLit : ""}`}
                  cx={w.x2}
                  cy={w.y2}
                  r="2.6"
                />
              </g>
            );
          })}
        </svg>

        {/* Layer bands */}
        <div className={styles.plane}>
          {BANDS.map((band, bandIndex) => {
            const nodes = NODES.filter((n) => n.band === bandIndex);
            const retrievalBand = bandIndex === 2;
            return (
              <div className={styles.band} key={band}>
                <span className={styles.bandLabel}>
                  <span className={styles.bandIndex}>
                    {String(bandIndex + 1).padStart(2, "0")}
                  </span>
                  {band}
                </span>
                <div className={styles.bandBody}>
                  {retrievalBand ? (
                    <>
                      <div className={styles.row}>{renderNode(nodes[0])}</div>
                      <div className={styles.forkRow}>
                        {renderNode(nodes[1])}
                        {renderNode(nodes[2])}
                      </div>
                      <div className={styles.row}>{renderNode(nodes[3])}</div>
                    </>
                  ) : (
                    // Sequential stages get their own row each — side-by-side
                    // would read as parallel branches and bend their wires.
                    nodes.map((n) => (
                      <div className={styles.row} key={n.id}>
                        {renderNode(n)}
                      </div>
                    ))
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
