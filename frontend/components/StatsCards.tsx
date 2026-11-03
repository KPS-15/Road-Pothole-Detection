"use client";

import React from "react";
import { AlertTriangle, CheckCircle, Film, Gauge, Clock } from "lucide-react";

interface StatsCardsProps {
  totalFrames: number;
  potholeFrames: number;
  normalFrames: number;
  avgLatencyMs: number;
  fps: number;
  processingTimeSec?: number;
  isLiveAnalysis?: boolean;
}

interface StatCardProps {
  label: string;
  value: string | number;
  sub: string;
  accent: string;
  accentDim: string;
  accentRim: string;
  icon: React.ReactNode;
  badge?: React.ReactNode;
}

function StatCard({ label, value, sub, accent, accentDim, accentRim, icon, badge }: StatCardProps) {
  return (
    <div
      className="p-4 flex flex-col gap-3 transition-all duration-150"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderLeft: `3px solid ${accent}`,
        borderRadius: "4px",
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0 1px ${accentRim}, 4px 0 24px -8px ${accentDim}`;
        (e.currentTarget as HTMLElement).style.borderColor = accentRim;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.boxShadow = "";
        (e.currentTarget as HTMLElement).style.borderColor = "var(--rim)";
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2" style={{ color: accent }}>
          {icon}
          <span className="section-label" style={{ color: "var(--muted)", letterSpacing: "0.1em" }}>
            {label}
          </span>
        </div>
        {badge}
      </div>

      <div className="stat-num text-3xl" style={{ color: "var(--ink)" }}>
        {value}
      </div>

      <div className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
        {sub}
      </div>
    </div>
  );
}

export default function StatsCards({
  totalFrames,
  potholeFrames,
  normalFrames,
  avgLatencyMs,
  fps,
  processingTimeSec,
  isLiveAnalysis = false,
}: StatsCardsProps) {
  const potholeRatio =
    totalFrames > 0 ? ((potholeFrames / totalFrames) * 100).toFixed(1) : "0.0";
  const normalRatio =
    totalFrames > 0 ? (100 - parseFloat(potholeRatio)).toFixed(1) : "0.0";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      <StatCard
        label="Frames Evaluated"
        value={totalFrames.toLocaleString()}
        sub={processingTimeSec ? `Total time: ${processingTimeSec}s` : totalFrames > 0 ? "Sampled from dashcam" : "No active session"}
        accent="#FFFFFF"
        accentDim="rgba(255,255,255,0.06)"
        accentRim="rgba(255,255,255,0.22)"
        icon={<Film className="w-3.5 h-3.5" />}
        badge={
          isLiveAnalysis ? (
            <span className="badge badge-amber animate-pulse-dot">● LIVE</span>
          ) : undefined
        }
      />

      <StatCard
        label="Pothole Hazards"
        value={potholeFrames.toLocaleString()}
        sub={potholeFrames > 0 ? `${potholeRatio}% of evaluated frames` : "No hazard frames detected"}
        accent="#E5E5E5"
        accentDim="rgba(255,255,255,0.06)"
        accentRim="rgba(255,255,255,0.22)"
        icon={<AlertTriangle className="w-3.5 h-3.5" />}
        badge={
          totalFrames > 0 ? (
            <span className="badge badge-hazard">{potholeRatio}%</span>
          ) : undefined
        }
      />

      <StatCard
        label="Normal Road"
        value={normalFrames.toLocaleString()}
        sub={normalFrames > 0 ? `${normalRatio}% intact asphalt` : "Awaiting video analysis"}
        accent="#FFFFFF"
        accentDim="rgba(255,255,255,0.06)"
        accentRim="rgba(255,255,255,0.22)"
        icon={<CheckCircle className="w-3.5 h-3.5" />}
        badge={
          totalFrames > 0 ? (
            <span className="badge badge-acid">{normalRatio}%</span>
          ) : undefined
        }
      />

      <StatCard
        label="Processing Speed"
        value={avgLatencyMs > 0 ? `${avgLatencyMs.toFixed(1)}ms` : "--"}
        sub={fps > 0 ? `${fps.toFixed(1)} FPS · InceptionV3` : "Avg latency per frame"}
        accent="#D4D4D4"
        accentDim="rgba(255,255,255,0.05)"
        accentRim="rgba(255,255,255,0.20)"
        icon={<Gauge className="w-3.5 h-3.5" />}
        badge={
          fps > 0 ? (
            <span className="badge badge-ice">{fps.toFixed(1)} FPS</span>
          ) : undefined
        }
      />
    </div>
  );
}
