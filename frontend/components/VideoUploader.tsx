"use client";

import React, { useState, useRef } from "react";
import { Upload, Film, AlertCircle, Check, Loader2, X } from "lucide-react";
import { uploadDashcamVideo } from "../lib/api";
import { VideoUploadResult } from "../lib/types";

interface VideoUploaderProps {
  onUploadSuccess: (result: VideoUploadResult, file: File) => void;
  isUploading: boolean;
  setIsUploading: (val: boolean) => void;
}

export default function VideoUploader({
  onUploadSuccess,
  isUploading,
  setIsUploading,
}: VideoUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const validateAndSetFile = (file: File) => {
    setErrorMsg(null);
    const validExts = [".mp4", ".avi", ".mov", ".mkv"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!validExts.includes(ext)) {
      setErrorMsg(`Unsupported format (${ext}). Upload MP4, AVI, MOV, or MKV.`);
      return;
    }
    if (file.size > 150 * 1024 * 1024) {
      setErrorMsg("File exceeds 150MB limit.");
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) validateAndSetFile(e.dataTransfer.files[0]);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) validateAndSetFile(e.target.files[0]);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const result = await uploadDashcamVideo(selectedFile);
      onUploadSuccess(result, selectedFile);
    } catch (err: any) {
      setErrorMsg(err.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className="space-y-4"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderTop: "2px solid var(--amber)",
        borderRadius: 4,
        padding: 24,
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Film className="w-4 h-4" style={{ color: "var(--amber)" }} />
          <div>
            <div className="text-sm font-bold text-ink">Upload Dashcam Video</div>
            <div className="font-mono text-[11px]" style={{ color: "var(--dim)" }}>
              MP4, AVI, MOV, MKV · Max 150MB
            </div>
          </div>
        </div>
        <span className="badge badge-amber">POTHOLE PIPELINE</span>
      </div>

      {/* Drop zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className="drop-zone"
        style={{
          padding: "40px 24px",
          textAlign: "center",
          cursor: "pointer",
          borderColor: dragActive
            ? "var(--amber)"
            : selectedFile
            ? "rgba(163,230,53,0.4)"
            : undefined,
          background: dragActive
            ? "var(--amber-glow)"
            : selectedFile
            ? "rgba(163,230,53,0.05)"
            : undefined,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
          onChange={handleChange}
          className="hidden"
        />

        {selectedFile ? (
          <div className="space-y-2">
            <div
              className="w-10 h-10 flex items-center justify-center mx-auto"
              style={{
                background: "rgba(163,230,53,0.12)",
                border: "1px solid rgba(163,230,53,0.3)",
                borderRadius: 3,
              }}
            >
              <Check className="w-5 h-5" style={{ color: "var(--acid)" }} />
            </div>
            <div className="text-sm font-bold text-ink">{selectedFile.name}</div>
            <div className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · Click to replace
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div
              className="w-10 h-10 flex items-center justify-center mx-auto"
              style={{
                background: "var(--amber-glow)",
                border: "1px solid var(--amber-rim)",
                borderRadius: 3,
              }}
            >
              <Upload className="w-5 h-5" style={{ color: "var(--amber)" }} />
            </div>
            <div>
              <p className="text-sm font-bold text-ink">
                Drag & drop or{" "}
                <span style={{ color: "var(--amber)", textDecoration: "underline" }}>
                  browse files
                </span>
              </p>
              <p className="font-mono text-[11px] mt-1" style={{ color: "var(--dim)" }}>
                Standard dashcam footage up to 1080p supported
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {errorMsg && (
        <div
          className="flex items-center gap-3 p-3"
          style={{
            background: "var(--hazard-d)",
            border: "1px solid var(--hazard-rim)",
            borderRadius: 3,
          }}
        >
          <AlertCircle className="w-4 h-4 shrink-0" style={{ color: "var(--hazard)" }} />
          <span className="font-mono text-[12px] text-white">
            {errorMsg}
          </span>
        </div>
      )}

      {/* Actions */}
      {selectedFile && (
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
            disabled={isUploading}
            className="btn btn-ghost"
            style={{ fontSize: 12 }}
          >
            <X className="w-3.5 h-3.5" /> Clear
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); handleUploadSubmit(); }}
            disabled={isUploading}
            className="btn btn-primary"
            style={{ fontSize: 12 }}
          >
            {isUploading ? (
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Uploading…</>
            ) : (
              <><Upload className="w-3.5 h-3.5" /> Upload & Run Inference</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
