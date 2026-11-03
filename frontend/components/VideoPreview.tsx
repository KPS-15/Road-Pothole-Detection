"use client";

import React, { useState, useMemo } from "react";
import { Play, RotateCcw, Sliders, Film, Loader2 } from "lucide-react";
import { VideoUploadResult } from "../lib/types";

interface VideoPreviewProps {
  videoUpload: VideoUploadResult;
  videoFile: File | null;
  onStartProcessing: (threshold: number, frameInterval: number, generateAnnotated: boolean) => void;
  isProcessing: boolean;
  onReset: () => void;
}

export default function VideoPreview({
  videoUpload,
  videoFile,
  onStartProcessing,
  isProcessing,
  onReset,
}: VideoPreviewProps) {
  const [threshold, setThreshold] = useState(0.50);
  const [frameInterval, setFrameInterval] = useState(3);
  const [generateAnnotated, setGenerateAnnotated] = useState(true);

  const videoObjectUrl = useMemo(() => {
    if (!videoFile) return null;
    return URL.createObjectURL(videoFile);
  }, [videoFile]);

  const ordinal = (n: number) => {
    if (n === 1) return "1st"; if (n === 2) return "2nd"; if (n === 3) return "3rd";
    return `${n}th`;
  };

  const metaCells = [
    { label: "Resolution", value: `${videoUpload.width}×${videoUpload.height}` },
    { label: "Duration",   value: `${videoUpload.duration_sec}s` },
    { label: "Frames",     value: videoUpload.total_frames.toLocaleString() },
    { label: "FPS",        value: `${videoUpload.fps}` },
    { label: "File Size",  value: `${(videoUpload.file_size_bytes / (1024 * 1024)).toFixed(2)} MB` },
  ];

  return (
    <div
      className="space-y-5"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderTop: "2px solid var(--amber)",
        borderRadius: 4,
        padding: 24,
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <Film className="w-4 h-4 shrink-0" style={{ color: "var(--amber)" }} />
          <div className="min-w-0">
            <div className="text-sm font-bold text-ink truncate">{videoUpload.original_filename}</div>
            <div className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
              Verified format · Ready for InceptionV3 frame analysis
            </div>
          </div>
        </div>
        <button onClick={onReset} disabled={isProcessing} className="btn btn-ghost shrink-0" style={{ fontSize: 12 }}>
          <RotateCcw className="w-3.5 h-3.5" /> Replace
        </button>
      </div>

      {/* Meta cells */}
      <div className="grid grid-cols-5 gap-2">
        {metaCells.map((cell, i) => (
          <div
            key={i}
            className="p-2.5 text-center"
            style={{ background: "var(--raised)", border: "1px solid var(--rim)", borderRadius: 3 }}
          >
            <div className="section-label mb-1">{cell.label}</div>
            <div className="stat-num text-sm text-ink">{cell.value}</div>
          </div>
        ))}
      </div>

      {/* Video preview */}
      <div
        className="overflow-hidden flex items-center justify-center"
        style={{
          background: "#000",
          border: "1px solid var(--rim)",
          borderRadius: 3,
          aspectRatio: "16/9",
          maxHeight: 360,
        }}
      >
        {videoObjectUrl ? (
          <video src={videoObjectUrl} controls style={{ width: "100%", height: "100%", objectFit: "contain" }} />
        ) : (
          <div className="font-mono text-[12px]" style={{ color: "var(--dim)" }}>Loading preview…</div>
        )}
      </div>

      {/* Config panel */}
      <div
        className="p-4 space-y-5"
        style={{ background: "var(--raised)", border: "1px solid var(--rim)", borderRadius: 3 }}
      >
        <div className="flex items-center gap-2">
          <Sliders className="w-3.5 h-3.5" style={{ color: "var(--amber)" }} />
          <span className="section-label">Inference Configuration</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Threshold */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="section-label">Decision Threshold</span>
              <span className="stat-num text-sm" style={{ color: "var(--amber)" }}>
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range" min="0.10" max="0.90" step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              disabled={isProcessing}
            />
            <p className="font-mono text-[10px]" style={{ color: "var(--dim)" }}>
              Probability threshold for triggering pothole alert per frame.
            </p>
          </div>

          {/* Frame interval */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="section-label">Sampling Stride</span>
              <span className="stat-num text-sm" style={{ color: "var(--amber)" }}>
                {ordinal(frameInterval)} frame
              </span>
            </div>
            <input
              type="range" min="1" max="15" step="1"
              value={frameInterval}
              onChange={(e) => setFrameInterval(parseInt(e.target.value))}
              disabled={isProcessing}
            />
            <p className="font-mono text-[10px]" style={{ color: "var(--dim)" }}>
              ~{Math.round(videoUpload.total_frames / frameInterval)} frames to evaluate. Higher = faster.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 pt-2" style={{ borderTop: "1px solid var(--rim)" }}>
          <input
            type="checkbox"
            id="annotated"
            checked={generateAnnotated}
            onChange={(e) => setGenerateAnnotated(e.target.checked)}
            disabled={isProcessing}
            style={{ width: 14, height: 14 }}
          />
          <label htmlFor="annotated" className="font-mono text-[12px] cursor-pointer" style={{ color: "var(--muted)" }}>
            Render HUD telemetry overlay and generate downloadable annotated video
          </label>
        </div>
      </div>

      {/* Start button */}
      <button
        onClick={() => onStartProcessing(threshold, frameInterval, generateAnnotated)}
        disabled={isProcessing}
        className="btn btn-primary w-full disabled:opacity-50"
        style={{ fontSize: 14, padding: "12px 20px" }}
      >
        {isProcessing ? (
          <><Loader2 className="w-4 h-4 animate-spin" /> Processing Frames…</>
        ) : (
          <><Play className="w-4 h-4" fill="currentColor" /> Start InceptionV3 Dashcam Inference</>
        )}
      </button>
    </div>
  );
}
