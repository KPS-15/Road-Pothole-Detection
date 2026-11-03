"use client";

import React from "react";
import {
  BarChart3,
  AlertCircle,
  FileText,
  TrendingUp,
  PieChart,
  Layers,
} from "lucide-react";
import { EvaluationMetrics } from "../lib/types";
import { getStaticArtifactUrl } from "../lib/api";

interface MetricsChartsProps {
  metrics: EvaluationMetrics | null;
  isLoading: boolean;
}

function MetricCell({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div
      className="p-4 space-y-1.5"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderTop: `2px solid ${color}`,
        borderRadius: 4,
      }}
    >
      <div className="section-label">{label}</div>
      <div className="stat-num text-2xl" style={{ color }}>
        {value}
      </div>
      <div className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
        {sub}
      </div>
    </div>
  );
}

function ArtifactPanel({
  title,
  src,
  alt,
  caption,
  icon,
  accentColor,
}: {
  title: string;
  src: string;
  alt: string;
  caption: string;
  icon: React.ReactNode;
  accentColor: string;
}) {
  return (
    <div
      className="space-y-3 p-4"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderLeft: `2px solid ${accentColor}`,
        borderRadius: 4,
      }}
    >
      <div className="flex items-center gap-2">
        {icon}
        <span className="section-label" style={{ color: "var(--muted)" }}>
          {title}
        </span>
      </div>
      <div
        className="overflow-hidden flex items-center justify-center"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--rim)",
          borderRadius: 3,
          aspectRatio: "16/9",
        }}
      >
        <img
          src={src}
          alt={alt}
          onError={(e) => {
            (e.target as HTMLElement).style.display = "none";
          }}
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
        />
      </div>
      <p className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
        {caption}
      </p>
    </div>
  );
}

export default function MetricsCharts({ metrics, isLoading }: MetricsChartsProps) {
  if (isLoading) {
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-4 gap-3">
          {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-24 rounded" />)}
        </div>
        <div className="grid grid-cols-2 gap-4">
          {[...Array(4)].map((_, i) => <div key={i} className="skeleton h-52 rounded" />)}
        </div>
      </div>
    );
  }

  const hasData = metrics?.has_results && metrics.summary?.overall_metrics;
  const om = metrics?.summary?.overall_metrics;

  return (
    <div className="space-y-5">
      {/* Metric cells */}
      {hasData ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCell
            label="Test Accuracy"
            value={`${(om!.accuracy * 100).toFixed(2)}%`}
            sub="Held-out test partition"
            color="#FFFFFF"
          />
          <MetricCell
            label="Pothole Precision"
            value={`${(om!.precision_pothole * 100).toFixed(2)}%`}
            sub="Positive class alert accuracy"
            color="#E5E5E5"
          />
          <MetricCell
            label="Pothole Recall"
            value={`${(om!.recall_pothole * 100).toFixed(2)}%`}
            sub="Road defect detection rate"
            color="#D4D4D4"
          />
          <MetricCell
            label="F1-Score"
            value={om!.f1_score_pothole.toFixed(4)}
            sub="Harmonic mean (P / R)"
            color="#FFFFFF"
          />
        </div>
      ) : (
        <div
          className="flex items-center gap-4 p-5"
          style={{
            background: "var(--card)",
            border: "1px solid var(--rim)",
            borderLeft: "3px solid #FFFFFF",
            borderRadius: 4,
          }}
        >
          <AlertCircle className="w-5 h-5 shrink-0" style={{ color: "#FFFFFF" }} />
          <div>
            <div className="text-sm font-bold text-ink">Evaluation Metrics Pending</div>
            <div className="font-mono text-[11px] mt-1" style={{ color: "var(--muted)" }}>
              Run <code style={{ color: "#FFFFFF" }}>python evaluate.py</code> to populate test set metrics.
            </div>
          </div>
        </div>
      )}

      {/* Artifact charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ArtifactPanel
          title="Training & Validation Convergence"
          src={getStaticArtifactUrl("results/training_curves.png")}
          alt="Training curves"
          caption="Cross-entropy loss & accuracy across frozen head and fine-tuning epochs."
          icon={<TrendingUp className="w-3.5 h-3.5 text-white" />}
          accentColor="#FFFFFF"
        />
        <ArtifactPanel
          title="Confusion Matrix Heatmap"
          src={getStaticArtifactUrl("results/confusion_matrix.png")}
          alt="Confusion matrix"
          caption="TP vs FP rates demonstrating low false-alert rate on normal asphalt."
          icon={<Layers className="w-3.5 h-3.5 text-white" />}
          accentColor="#D4D4D4"
        />
        <ArtifactPanel
          title="ROC & Precision-Recall Curves"
          src={getStaticArtifactUrl("results/roc_pr_curves.png")}
          alt="ROC and PR curves"
          caption="AUC score verifies discriminatory power across all threshold configurations."
          icon={<BarChart3 className="w-3.5 h-3.5 text-white" />}
          accentColor="#E5E5E5"
        />
        <ArtifactPanel
          title="Class Distribution & Dataset Partition"
          src={getStaticArtifactUrl("results/dataset_class_distribution.png")}
          alt="Dataset distribution"
          caption="Stratified 70/15/15 split: Train / Validation / Held-out Test partitions."
          icon={<PieChart className="w-3.5 h-3.5 text-white" />}
          accentColor="#CCCCCC"
        />
      </div>

      {/* Classification report */}
      {metrics?.classification_report && (
        <div
          className="p-5 space-y-3"
          style={{ background: "var(--card)", border: "1px solid var(--rim)", borderRadius: 4 }}
        >
          <div className="flex items-center gap-2">
            <FileText className="w-3.5 h-3.5" style={{ color: "var(--amber)" }} />
            <span className="section-label">Classification Report</span>
          </div>
          <pre
            className="font-mono text-[12px] leading-relaxed overflow-x-auto"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--rim)",
              borderRadius: 3,
              padding: "16px",
              color: "var(--muted)",
              whiteSpace: "pre-wrap",
            }}
          >
            {metrics.classification_report}
          </pre>
        </div>
      )}
    </div>
  );
}
