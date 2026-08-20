import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ThemeToggle from "@/components/ThemeToggle";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "RAG Intelligence — Production-Grade Retrieval System",
  description:
    "A production-grade RAG system using LangChain, Qdrant, and gpt-oss-120b. Hybrid search with cross-encoder re-ranking and strict citation enforcement.",
  keywords: ["RAG", "LangChain", "Qdrant", "AI", "LLM", "Retrieval"],
};

/* Runs before first paint, so the correct theme is on <html> when the page
   renders — otherwise every load flashes dark before switching to light. */
const themeScript = `
(function () {
  try {
    var stored = localStorage.getItem('rag-theme');
    var system = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    document.documentElement.dataset.theme = stored || system;
  } catch (e) {
    document.documentElement.dataset.theme = 'dark';
  }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}
