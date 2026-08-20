"use client";

import { useEffect, useState } from "react";
import styles from "./HealthBadge.module.css";

type State = "checking" | "ok" | "degraded" | "down";

const LABEL: Record<State, string> = {
  checking: "Checking backend…",
  ok: "System Operational",
  degraded: "Degraded — vector store unreachable",
  down: "Backend Unreachable",
};

/**
 * Reports the backend's real state. The footer used to claim "System
 * Operational" unconditionally, which stayed green even when the API was down.
 */
export default function HealthBadge() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`, {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(String(res.status));
        const body = await res.json();
        if (!cancelled) setState(body.qdrant_connected ? "ok" : "degraded");
      } catch {
        if (!cancelled) setState("down");
      }
    };

    check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <span className={`${styles.badge} ${styles[state]}`} role="status">
      <span className={styles.dot} />
      {LABEL[state]}
    </span>
  );
}
