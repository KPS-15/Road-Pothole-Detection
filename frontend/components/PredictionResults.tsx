"use client";

import React, { useState } from "react";
import {
  Download,
  RotateCcw,
  AlertTriangle,
  CheckCircle,
  Film,
  Camera,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { VideoResults, FramePrediction } from "../lib/types";
import { getVideoDownloadUrl } from "../lib/api";

interface PredictionResultsProps {
  results: VideoResults;
  onReset: () => void;
}

export default function PredictionResults({
  results,
  onReset,
}: PredictionResultsProps) {
  const [filterMode, setFilterMode] = useState<"all" | "pothole" | "normal">("all");
  const [selectedSnapshot, setSelectedSnapshot] = useState<FramePrediction | null>(
    results.top_pothole_frames.length > 0 ? results.top_pothole_frames[0] : null
  );

  const filteredFrames = results.frame_timeline.filter((f) => {
    if (filterMode === "pothole") return f.is_pothole;
    if (filterMode === "normal") return !f.is_pothole;
    return true;
  });

  const chartData = results.frame_timeline.map((f) => ({
    frame: f.frame_number,
    time: f.timestamp_sec,
    prob: Math.round(f.pothole_probability * 100),
    isPothole: f.is_pothole,
  }));

  const downloadUrl = getVideoDownloadUrl(results.job_id);

  return (
    <div className="space-y-4 animate-fade-up">
      {/* Header Banner */}
      <div
        className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4"
        style={{
          background: "var(--card)",
          border: "1px solid var(--rim)",
          borderLeft: "3px solid #FFFFFF",
          borderRadius: 4,
        }}
      >
        <div>
          <div className="flex items-center gap-2">
            <span className="badge badge-amber">ANALYSIS COMPLETE</span>
            <h2 className="text-base font-bold text-white tracking-tight">
              {results.original_filename}
            </h2>
          </div>
          <p className="font-mono text-[11px] mt-1" style={{ color: "var(--muted)" }}>
            Processed {results.evaluated_frames_count} sampled frames in {results.total_processing_time_sec}s ({results.estimated_fps} FPS throughput)
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {results.has_annotated_video && (
            <a
              href={downloadUrl}
              download={`annotated_${results.original_filename}`}
              className="btn btn-primary"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Annotated MP4</span>
            </a>
          )}

          <button
            onClick={onReset}
            className="btn btn-ghost"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Analyse Another</span>
          </button>
        </div>
      </div>

      {/* Severity Timeline Chart */}
      <div
        className="p-5 space-y-4"
        style={{
          background: "var(--card)",
          border: "1px solid var(--rim)",
          borderRadius: 4,
        }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
              <Film className="w-4 h-4 text-white" />
              Road Hazard Severity Timeline
            </h3>
            <p className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              InceptionV3 Pothole Probability (%) across dashcam playback time
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-white">
              <span className="w-2.5 h-2.5 rounded-full bg-white inline-block" /> &gt; 50% Distress Zone
            </span>
            <span className="flex items-center gap-1.5" style={{ color: "var(--dim)" }}>
              <span className="w-2.5 h-2.5 rounded-full bg-neutral-600 inline-block" /> Normal Road
            </span>
          </div>
        </div>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="potholeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FFFFFF" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#FFFFFF" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="time"
                stroke="#666666"
                fontSize={11}
                tickFormatter={(val) => `${val}s`}
              />
              <YAxis
                domain={[0, 100]}
                stroke="#666666"
                fontSize={11}
                tickFormatter={(val) => `${val}%`}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div
                        className="p-2.5 text-xs font-mono"
                        style={{
                          background: "#0A0A0A",
                          border: "1px solid #333333",
                          borderRadius: 3,
                          color: "#FFFFFF",
                        }}
                      >
                        <div className="text-white font-bold mb-1">
                          Frame {data.frame} ({data.time}s)
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className={data.isPothole ? "text-white font-bold" : "text-neutral-400"}>
                            {data.isPothole ? "POTHOLE" : "NORMAL ROAD"}
                          </span>
                          <span>• {data.prob}%</span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine
                y={50}
                stroke="#888888"
                strokeDasharray="4 4"
                label={{ value: "50% Threshold", fill: "#888888", fontSize: 10, position: "insideTopRight" }}
              />
              <Area
                type="monotone"
                dataKey="prob"
                stroke="#FFFFFF"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#potholeGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Defect Snapshots Gallery */}
      {results.top_pothole_frames.length > 0 && (
        <div
          className="p-5 space-y-4"
          style={{
            background: "var(--card)",
            border: "1px solid var(--rim)",
            borderRadius: 4,
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Camera className="w-4 h-4 text-white" />
                Detected Pothole Snapshots
              </h3>
              <p className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
                High-confidence road surface distress zones captured by HUD
              </p>
            </div>
            <span className="badge badge-hazard">
              {results.pothole_frames_count} Hazard Frames
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {results.top_pothole_frames.slice(0, 8).map((snap, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedSnapshot(snap)}
                className="cursor-pointer overflow-hidden transition-all"
                style={{
                  background: "var(--raised)",
                  border: selectedSnapshot?.frame_number === snap.frame_number
                    ? "1px solid #FFFFFF"
                    : "1px solid var(--rim)",
                  borderRadius: 3,
                }}
              >
                {snap.thumbnail_data_url ? (
                  <img
                    src={snap.thumbnail_data_url}
                    alt={`Frame ${snap.frame_number}`}
                    className="w-full aspect-video object-cover"
                  />
                ) : (
                  <div
                    className="w-full aspect-video flex items-center justify-center font-mono text-xs"
                    style={{ background: "#000", color: "var(--dim)" }}
                  >
                    Frame {snap.frame_number}
                  </div>
                )}
                <div
                  className="p-2 font-mono text-[11px] flex items-center justify-between"
                  style={{ background: "var(--panel)" }}
                >
                  <span className="text-white font-medium">Frame {snap.frame_number}</span>
                  <span className="text-white font-bold">
                    {(snap.pothole_probability * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          {selectedSnapshot && selectedSnapshot.thumbnail_data_url && (
            <div
              className="p-4 flex flex-col md:flex-row gap-4 items-center"
              style={{
                background: "var(--raised)",
                border: "1px solid var(--rim)",
                borderRadius: 4,
              }}
            >
              <img
                src={selectedSnapshot.thumbnail_data_url}
                alt="Enlarged snapshot"
                className="w-full md:w-1/2 aspect-video object-contain bg-black"
                style={{ border: "1px solid var(--rim)", borderRadius: 3 }}
              />
              <div className="flex-1 space-y-2 text-xs">
                <div className="text-white font-bold text-sm flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4 text-white" /> Road Pothole Detected
                </div>
                <div className="grid grid-cols-2 gap-2 font-mono pt-2">
                  <div className="p-2" style={{ background: "var(--surface)", border: "1px solid var(--rim)", borderRadius: 3 }}>
                    <span className="block text-[10px]" style={{ color: "var(--dim)" }}>FRAME</span>
                    <span className="text-white font-bold">{selectedSnapshot.frame_number}</span>
                  </div>
                  <div className="p-2" style={{ background: "var(--surface)", border: "1px solid var(--rim)", borderRadius: 3 }}>
                    <span className="block text-[10px]" style={{ color: "var(--dim)" }}>TIME OFFSET</span>
                    <span className="text-white font-bold">{selectedSnapshot.timestamp_sec}s</span>
                  </div>
                  <div className="p-2" style={{ background: "var(--surface)", border: "1px solid var(--rim)", borderRadius: 3 }}>
                    <span className="block text-[10px]" style={{ color: "var(--dim)" }}>CONFIDENCE</span>
                    <span className="text-white font-bold">
                      {(selectedSnapshot.pothole_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-2" style={{ background: "var(--surface)", border: "1px solid var(--rim)", borderRadius: 3 }}>
                    <span className="block text-[10px]" style={{ color: "var(--dim)" }}>INFERENCE TIME</span>
                    <span className="text-white font-bold">
                      {selectedSnapshot.inference_time_ms} ms
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Processed Frames Log Table */}
      <div
        className="p-5 space-y-4"
        style={{
          background: "var(--card)",
          border: "1px solid var(--rim)",
          borderRadius: 4,
        }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Detailed Frame Classification Log
            </h3>
            <p className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              Showing {filteredFrames.length} of {results.evaluated_frames_count} evaluated frames
            </p>
          </div>

          <div
            className="flex items-center gap-1 p-1 font-mono text-xs"
            style={{ background: "var(--surface)", border: "1px solid var(--rim)", borderRadius: 3 }}
          >
            <button
              onClick={() => setFilterMode("all")}
              className="px-3 py-1 rounded transition-all"
              style={{
                background: filterMode === "all" ? "rgba(255,255,255,0.12)" : "transparent",
                color: filterMode === "all" ? "#FFFFFF" : "var(--muted)",
                fontWeight: filterMode === "all" ? 600 : 400,
              }}
            >
              All ({results.evaluated_frames_count})
            </button>
            <button
              onClick={() => setFilterMode("pothole")}
              className="px-3 py-1 rounded transition-all"
              style={{
                background: filterMode === "pothole" ? "rgba(255,255,255,0.12)" : "transparent",
                color: filterMode === "pothole" ? "#FFFFFF" : "var(--muted)",
                fontWeight: filterMode === "pothole" ? 600 : 400,
              }}
            >
              Potholes ({results.pothole_frames_count})
            </button>
            <button
              onClick={() => setFilterMode("normal")}
              className="px-3 py-1 rounded transition-all"
              style={{
                background: filterMode === "normal" ? "rgba(255,255,255,0.12)" : "transparent",
                color: filterMode === "normal" ? "#FFFFFF" : "var(--muted)",
                fontWeight: filterMode === "normal" ? 600 : 400,
              }}
            >
              Normal ({results.normal_frames_count})
            </button>
          </div>
        </div>

        <div
          className="overflow-x-auto max-h-72 overflow-y-auto"
          style={{ border: "1px solid var(--rim)", borderRadius: 3 }}
        >
          <table className="w-full text-left text-xs font-mono">
            <thead
              className="sticky top-0 text-ink"
              style={{ background: "var(--raised)", borderBottom: "1px solid var(--rim)" }}
            >
              <tr>
                <th className="p-3">Frame #</th>
                <th className="p-3">Timestamp</th>
                <th className="p-3">Classification</th>
                <th className="p-3">Pothole Probability</th>
                <th className="p-3">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: "var(--rim)", color: "#FFFFFF" }}>
              {filteredFrames.map((f) => (
                <tr
                  key={f.frame_number}
                  className="transition-colors hover:bg-white/5"
                >
                  <td className="p-3 font-semibold">{f.frame_number}</td>
                  <td className="p-3" style={{ color: "var(--dim)" }}>{f.timestamp_sec}s</td>
                  <td className="p-3">
                    {f.is_pothole ? (
                      <span className="badge badge-hazard">
                        <AlertTriangle className="w-3 h-3 text-white" /> POTHOLE
                      </span>
                    ) : (
                      <span className="badge badge-acid">
                        <CheckCircle className="w-3 h-3 text-white" /> NORMAL
                      </span>
                    )}
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-16 h-1.5 overflow-hidden"
                        style={{ background: "var(--rim2)", borderRadius: 1 }}
                      >
                        <div
                          className="h-full bg-white"
                          style={{ width: `${Math.round(f.pothole_probability * 100)}%` }}
                        />
                      </div>
                      <span>{(f.pothole_probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="p-3" style={{ color: "var(--dim)" }}>{f.inference_time_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
