"use client";

import React from "react";
import { Cpu, CheckCircle2, AlertCircle, Layers, Sliders, HardDrive } from "lucide-react";
import { ModelStatus as ModelStatusType } from "../lib/types";

interface ModelStatusProps {
  status: ModelStatusType | null;
  isLoading: boolean;
}

export default function ModelStatus({ status, isLoading }: ModelStatusProps) {
  if (isLoading) {
    return (
      <div className="panel p-5 space-y-3" style={{ borderRadius: 4 }}>
        <div className="skeleton h-4 w-40" />
        <div className="grid grid-cols-4 gap-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton h-14 rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div
        className="panel p-5 flex items-center gap-4"
        style={{ borderLeft: "3px solid var(--hazard)", borderRadius: 4 }}
      >
        <AlertCircle className="w-5 h-5 shrink-0" style={{ color: "var(--hazard)" }} />
        <div>
          <div className="text-sm font-bold text-ink">Backend Offline</div>
          <div className="font-mono text-[11px] mt-1" style={{ color: "var(--muted)" }}>
            Cannot reach FastAPI server at http://127.0.0.1:8000
          </div>
        </div>
      </div>
    );
  }

  const specs = [
    {
      label: "Input Shape",
      value: status.input_shape.join(" × "),
      sub: "RGB · Normalized [-1, 1]",
      icon: <Layers className="w-3.5 h-3.5" style={{ color: "#FFFFFF" }} />,
    },
    {
      label: "Parameters",
      value: status.total_params ? `${(status.total_params / 1e6).toFixed(1)}M` : "~21.8M",
      sub: "Backbone + Head",
      icon: <HardDrive className="w-3.5 h-3.5" style={{ color: "#D4D4D4" }} />,
    },
    {
      label: "Dropout Rate",
      value: status.config?.dropout_rate ?? 0.4,
      sub: "EarlyStopping active",
      icon: <Sliders className="w-3.5 h-3.5" style={{ color: "#FFFFFF" }} />,
    },
    {
      label: "Learning Rate",
      value: status.config?.head_learning_rate ?? "1e-3",
      sub: `Fine-tune: ${status.config?.fine_tune_learning_rate ?? "1e-5"}`,
      icon: <Cpu className="w-3.5 h-3.5" style={{ color: "#D4D4D4" }} />,
    },
  ];

  return (
    <div
      className="overflow-hidden"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderLeft: "3px solid var(--amber)",
        borderRadius: 4,
      }}
    >
      {/* Header row */}
      <div
        className="px-5 py-3 flex items-center justify-between"
        style={{ borderBottom: "1px solid var(--rim)", background: "var(--raised)" }}
      >
        <div className="flex items-center gap-3">
          <Cpu className="w-4 h-4" style={{ color: "var(--amber)" }} />
          <div>
            <div className="text-sm font-bold text-ink">InceptionV3 Model</div>
            <div className="font-mono text-[10px]" style={{ color: "var(--dim)" }}>
              ImageNet pretrained · Transfer learning
            </div>
          </div>
        </div>

        {status.is_loaded ? (
          <span
            className="status-pill"
            style={{
              background: "rgba(255,255,255,0.08)",
              color: "#FFFFFF",
              border: "1px solid rgba(255,255,255,0.22)",
            }}
          >
            <CheckCircle2 className="w-3 h-3" /> Loaded & Operational
          </span>
        ) : status.is_available ? (
          <span
            className="status-pill"
            style={{
              background: "rgba(255,255,255,0.05)",
              color: "#D4D4D4",
              border: "1px solid rgba(255,255,255,0.18)",
            }}
          >
            <AlertCircle className="w-3 h-3" /> Checkpoint Ready
          </span>
        ) : (
          <span
            className="status-pill"
            style={{
              background: "rgba(255,255,255,0.03)",
              color: "var(--dim)",
              border: "1px solid var(--rim)",
            }}
          >
            <AlertCircle className="w-3 h-3" /> Checkpoint Missing
          </span>
        )}
      </div>

      {/* Spec cells */}
      <div className="grid grid-cols-2 md:grid-cols-4">
        {specs.map((spec, i) => (
          <div
            key={i}
            className="px-4 py-3 space-y-1.5"
            style={{
              borderRight: i < 3 ? "1px solid var(--rim)" : undefined,
            }}
          >
            <div className="flex items-center gap-1.5 section-label">
              {spec.icon}
              {spec.label}
            </div>
            <div className="stat-num text-base text-ink">{spec.value}</div>
            <div className="font-mono text-[10px]" style={{ color: "var(--dim)" }}>
              {spec.sub}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
