"use client";

import React, { useState } from "react";
import { Camera, Upload, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { predictSingleImage } from "../lib/api";
import { ImagePrediction } from "../lib/types";

export default function ImageTester() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.50);
  const [prediction, setPrediction] = useState<ImagePrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setErrorMsg(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedImage) return;
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await predictSingleImage(selectedImage, threshold);
      setPrediction(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to classify image.");
    } finally {
      setIsLoading(false);
    }
  };

  const isHazard = prediction?.is_hazard;

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
      <div className="flex items-center gap-3">
        <Camera className="w-4 h-4" style={{ color: "var(--amber)" }} />
        <div>
          <div className="text-sm font-bold text-ink">Single Frame Classifier</div>
          <div className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
            Upload a road surface image · InceptionV3 instant inference
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: upload + controls */}
        <div className="space-y-4">
          <label
            className="drop-zone block cursor-pointer"
            style={{ padding: previewUrl ? 0 : "32px 16px", textAlign: "center" }}
          >
            <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected road frame"
                style={{
                  width: "100%",
                  aspectRatio: "16/9",
                  objectFit: "contain",
                  background: "#000",
                  borderRadius: 3,
                  display: "block",
                }}
              />
            ) : (
              <div className="space-y-2">
                <div
                  className="w-9 h-9 flex items-center justify-center mx-auto"
                  style={{ background: "var(--amber-glow)", border: "1px solid var(--amber-rim)", borderRadius: 3 }}
                >
                  <Upload className="w-4 h-4" style={{ color: "var(--amber)" }} />
                </div>
                <p className="text-sm font-bold text-ink">Click to select image</p>
                <p className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
                  JPG, PNG, WebP · Resized to 299×299
                </p>
              </div>
            )}
          </label>

          {/* Threshold */}
          <div
            className="p-4 space-y-3"
            style={{ background: "var(--raised)", border: "1px solid var(--rim)", borderRadius: 3 }}
          >
            <div className="flex justify-between items-center">
              <span className="section-label">Decision Threshold</span>
              <span className="stat-num text-base" style={{ color: "var(--amber)" }}>
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
            <div className="flex justify-between font-mono text-[10px]" style={{ color: "var(--dim)" }}>
              <span>10% — Sensitive</span>
              <span>90% — Strict</span>
            </div>
          </div>

          <button
            onClick={handleRunInference}
            disabled={!selectedImage || isLoading}
            className="btn btn-primary w-full disabled:opacity-40"
            style={{ fontSize: 13 }}
          >
            {isLoading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Classifying…</>
            ) : (
              <><Camera className="w-4 h-4" /> Run InceptionV3 Inference</>
            )}
          </button>

          {errorMsg && (
            <div
              className="flex items-center gap-2 p-3"
              style={{ background: "var(--hazard-d)", border: "1px solid var(--hazard-rim)", borderRadius: 3 }}
            >
              <AlertTriangle className="w-4 h-4 shrink-0" style={{ color: "var(--hazard)" }} />
              <span className="font-mono text-[12px]" style={{ color: "#FC8181" }}>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Right: result */}
        <div className="flex flex-col justify-center">
          {prediction ? (
            <div
              className="space-y-5 p-5"
              style={{
                background: isHazard ? "rgba(239,68,68,0.05)" : "rgba(163,230,53,0.05)",
                border: `1px solid ${isHazard ? "rgba(239,68,68,0.3)" : "rgba(163,230,53,0.3)"}`,
                borderLeft: `3px solid ${isHazard ? "var(--hazard)" : "var(--acid)"}`,
                borderRadius: 4,
              }}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 flex items-center justify-center"
                  style={{
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.22)",
                    borderRadius: 3,
                  }}
                >
                  {isHazard ? (
                    <AlertTriangle className="w-5 h-5 text-white" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-white" />
                  )}
                </div>
                <div>
                  <div className="section-label">InceptionV3 Classification</div>
                  <div
                    className="stat-num text-xl mt-1 text-white"
                  >
                    {isHazard ? "POTHOLE DETECTED" : "NORMAL ROAD"}
                  </div>
                </div>
              </div>

              {/* Confidence bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <span className="section-label">Distress Probability</span>
                  <span className="stat-num text-sm text-white">
                    {(prediction.pothole_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="progress-bar" style={{ height: 4 }}>
                  <div
                    className="progress-fill"
                    style={{
                      width: `${prediction.pothole_probability * 100}%`,
                      background: "#FFFFFF",
                    }}
                  />
                </div>
              </div>

              {/* Info cells */}
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: "Confidence", value: `${prediction.confidence_percentage}%`, color: "var(--ink)" },
                  { label: "Inference", value: `${prediction.inference_time_ms} ms`, color: "var(--amber)" },
                ].map((item, i) => (
                  <div
                    key={i}
                    className="p-3"
                    style={{ background: "var(--raised)", border: "1px solid var(--rim)", borderRadius: 3 }}
                  >
                    <div className="section-label mb-1">{item.label}</div>
                    <div className="stat-num text-base" style={{ color: item.color }}>{item.value}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div
              className="flex flex-col items-center justify-center gap-3 p-10"
              style={{ background: "var(--raised)", border: "1px dashed var(--rim2)", borderRadius: 4 }}
            >
              <Camera className="w-8 h-8" style={{ color: "var(--dim)", opacity: 0.5 }} />
              <p className="font-mono text-[11px] text-center" style={{ color: "var(--dim)" }}>
                Upload an image and run inference<br />to see classification results here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
