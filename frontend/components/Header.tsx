"use client";

import React from "react";
import { RefreshCw } from "lucide-react";

interface HeaderProps {
  title: string;
  subtitle: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export default function Header({
  title,
  subtitle,
  onRefresh,
  isRefreshing = false,
}: HeaderProps) {
  return (
    <header
      className="h-14 px-6 sticky top-0 z-20 flex items-center justify-between"
      style={{
        background: "rgba(13,17,23,0.92)",
        borderBottom: "1px solid var(--rim)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center gap-4 min-w-0">
        {/* Page accent line */}
        <div
          className="h-7 w-0.5 rounded-full shrink-0"
          style={{ background: "var(--amber)" }}
        />
        <div className="min-w-0">
          <h1 className="text-sm font-bold text-ink tracking-tight truncate">
            {title}
          </h1>
          <p className="font-mono text-[10px] mt-px truncate" style={{ color: "var(--dim)", letterSpacing: "0.03em" }}>
            {subtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {/* Model badge */}
        <div
          className="hidden sm:flex items-center gap-2 px-2.5 py-1 font-mono text-[10px] font-bold tracking-wider"
          style={{
            background: "var(--amber-glow)",
            border: "1px solid var(--amber-rim)",
            borderRadius: "2px",
            color: "var(--amber-light)",
          }}
        >
          <span
            className="animate-pulse-dot inline-block"
            style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--amber)" }}
          />
          299×299 · INCEPTIONV3
        </div>

        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="btn btn-ghost"
            style={{ padding: "6px 12px", fontSize: 12 }}
            title="Refresh system state"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`}
              style={{ color: isRefreshing ? "var(--amber)" : undefined }}
            />
            <span>Sync</span>
          </button>
        )}
      </div>
    </header>
  );
}
