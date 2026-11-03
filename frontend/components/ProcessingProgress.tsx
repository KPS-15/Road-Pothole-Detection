"use client";

import React from "react";
import { Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { VideoStatus } from "../lib/types";

interface ProcessingProgressProps {
  status: VideoStatus;
}

export default function ProcessingProgress({ status }: ProcessingProgressProps) {
  const isCompleted = status.status === "completed";
  const isFailed = status.status === "failed";
  const pct = Math.max(0, Math.min(100, status.progress_percentage));
  const potholeRatio =
    status.processed_frames > 0
      ? ((status.pothole_count / status.processed_frames) * 100).toFixed(1)
      : "0.0";

  const stateColor = isCompleted ? "#FFFFFF" : isFailed ? "#A3A3A3" : "#FFFFFF";
  const stateDim   = "rgba(255,255,255,0.06)";
  const stateRim   = "rgba(255,255,255,0.22)";

  return (
    <div
      className="space-y-4"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderLeft: `3px solid ${stateColor}`,
        borderRadius: 4,
        padding: 20,
      }}
    >
      {/* Status row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 flex items-center justify-center"
            style={{ background: stateDim, border: `1px solid ${stateRim}`, borderRadius: 3 }}
          >
            {isCompleted ? (
              <CheckCircle2 className="w-4 h-4" style={{ color: stateColor }} />
            ) : isFailed ? (
              <AlertCircle className="w-4 h-4" style={{ color: stateColor }} />
            ) : (
              <Loader2 className="w-4 h-4 animate-spin" style={{ color: stateColor }} />
            )}
          </div>
          <div>
            <div className="text-sm font-bold text-ink">
              {isCompleted ? "Inference Complete" : isFailed ? "Processing Error" : "Analyzing Frames…"}
            </div>
            <div className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              {isCompleted
                ? "All sampled frames classified and summarized"
                : isFailed
                ? status.error || "An error occurred during inference"
                : `Frame ${status.current_frame} / ${status.total_frames}`}
            </div>
          </div>
        </div>

        <div
          className="stat-num text-2xl"
          style={{ color: stateColor }}
        >
          {pct.toFixed(0)}%
        </div>
      </div>

      {/* Progress bar */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${pct}%`,
            background: "#FFFFFF",
          }}
        />
      </div>

      {/* Live counts */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "Evaluated", value: status.processed_frames, color: "#FFFFFF" },
          { label: "Potholes", value: `${status.pothole_count} (${potholeRatio}%)`, color: "#E5E5E5" },
          { label: "Normal", value: status.normal_count, color: "#D4D4D4" },
        ].map((item, i) => (
          <div
            key={i}
            className="text-center py-2.5"
            style={{ background: "var(--raised)", border: "1px solid var(--rim)", borderRadius: 3 }}
          >
            <div className="section-label mb-1">{item.label}</div>
            <div className="stat-num text-sm" style={{ color: item.color }}>
              {item.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
