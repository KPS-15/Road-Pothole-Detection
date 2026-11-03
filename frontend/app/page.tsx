"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import StatsCards from "../components/StatsCards";
import ModelStatus from "../components/ModelStatus";
import VideoUploader from "../components/VideoUploader";
import VideoPreview from "../components/VideoPreview";
import ProcessingProgress from "../components/ProcessingProgress";
import PredictionResults from "../components/PredictionResults";
import MetricsCharts from "../components/MetricsCharts";
import ImageTester from "../components/ImageTester";

import {
  NavPage,
  ModelStatus as ModelStatusType,
  VideoUploadResult,
  VideoStatus,
  VideoResults,
  EvaluationMetrics,
} from "../lib/types";
import {
  fetchHealth,
  fetchModelStatus,
  fetchEvaluationMetrics,
  startVideoProcessing,
  fetchVideoStatus,
  fetchVideoResults,
} from "../lib/api";
import {
  ArrowRight,
  Film,
  Camera,
  BarChart3,
  Cpu,
  BookOpen,
  Info,
  Layers,
  TrendingUp,
} from "lucide-react";

const PAGE_META: Record<NavPage, { title: string; subtitle: string }> = {
  dashboard:       { title: "Mission Dashboard",          subtitle: "Overview · Telemetry · Quick actions" },
  "video-analysis":{ title: "Dashcam Video Analysis",     subtitle: "Upload footage for frame-by-frame pothole inference" },
  "image-test":    { title: "Single Frame Classifier",    subtitle: "Instant road surface snapshot test" },
  evaluation:      { title: "Model Evaluation Metrics",   subtitle: "Test set performance · Confusion matrix · ROC/PR curves" },
  about:           { title: "Architecture & Methodology", subtitle: "InceptionV3 transfer learning · Academic references" },
};

export default function Home() {
  const [activePage, setActivePage] = useState<NavPage>("dashboard");
  const [backendOnline, setBackendOnline] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatusType | null>(null);
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetrics | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadedVideo, setUploadedVideo] = useState<VideoUploadResult | null>(null);
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<VideoStatus | null>(null);
  const [videoResults, setVideoResults] = useState<VideoResults | null>(null);

  const pollingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const loadSystemData = async () => {
    setIsRefreshing(true);
    try {
      const [hRes, mRes, eRes] = await Promise.allSettled([
        fetchHealth(),
        fetchModelStatus(),
        fetchEvaluationMetrics(),
      ]);
      setBackendOnline(hRes.status === "fulfilled");
      if (mRes.status === "fulfilled") setModelStatus(mRes.value);
      if (eRes.status === "fulfilled") setEvaluationMetrics(eRes.value);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingStatus(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    loadSystemData();
    const interval = setInterval(loadSystemData, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!isProcessing || !uploadedVideo) {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
      return;
    }
    const checkStatus = async () => {
      try {
        const status = await fetchVideoStatus(uploadedVideo.job_id);
        setProcessingStatus(status);
        if (status.status === "completed") {
          setIsProcessing(false);
          const results = await fetchVideoResults(uploadedVideo.job_id);
          setVideoResults(results);
          clearInterval(pollingTimerRef.current!);
        } else if (status.status === "failed") {
          setIsProcessing(false);
          clearInterval(pollingTimerRef.current!);
        }
      } catch (err) {
        console.error(err);
      }
    };
    pollingTimerRef.current = setInterval(checkStatus, 1200);
    checkStatus();
    return () => { if (pollingTimerRef.current) clearInterval(pollingTimerRef.current); };
  }, [isProcessing, uploadedVideo]);

  const handleUploadSuccess = (result: VideoUploadResult, file: File) => {
    setUploadedVideo(result);
    setUploadedVideoFile(file);
    setVideoResults(null);
    setProcessingStatus(null);
  };

  const handleStartProcessing = async (threshold: number, frameInterval: number, generateAnnotated: boolean) => {
    if (!uploadedVideo) return;
    setIsProcessing(true);
    setVideoResults(null);
    try {
      const initialStatus = await startVideoProcessing(uploadedVideo.job_id, threshold, frameInterval, generateAnnotated);
      setProcessingStatus(initialStatus);
    } catch (err: any) {
      setIsProcessing(false);
      alert(err.message || "Failed to start video processing");
    }
  };

  const handleResetVideo = () => {
    setUploadedVideo(null);
    setUploadedVideoFile(null);
    setIsProcessing(false);
    setProcessingStatus(null);
    setVideoResults(null);
  };

  const meta = PAGE_META[activePage];

  return (
    <div className="flex min-h-screen" style={{ background: "var(--void)" }}>
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        modelStatus={modelStatus}
        backendOnline={backendOnline}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Header
          title={meta.title}
          subtitle={meta.subtitle}
          onRefresh={loadSystemData}
          isRefreshing={isRefreshing}
        />

        <main className="p-6 space-y-6 flex-1 max-w-6xl w-full mx-auto animate-fade-up">

          {/* ── DASHBOARD ─────────────────────────────────────────── */}
          {activePage === "dashboard" && (
            <div className="space-y-6">
              {/* Stats row */}
              <StatsCards
                totalFrames={videoResults?.evaluated_frames_count ?? (processingStatus?.processed_frames ?? 0)}
                potholeFrames={videoResults?.pothole_frames_count ?? (processingStatus?.pothole_count ?? 0)}
                normalFrames={videoResults?.normal_frames_count ?? (processingStatus?.normal_count ?? 0)}
                avgLatencyMs={videoResults?.average_latency_ms ?? 0}
                fps={videoResults?.estimated_fps ?? 0}
                processingTimeSec={videoResults?.total_processing_time_sec}
                isLiveAnalysis={isProcessing}
              />

              {/* Model status */}
              <ModelStatus status={modelStatus} isLoading={isLoadingStatus} />

              {/* CTA grid */}
              <div>
                <p className="section-label mb-3">Quick Actions</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <CTACard
                    icon={<Film className="w-4 h-4" />}
                    accent="#FFFFFF"
                    accentDim="rgba(255,255,255,0.06)"
                    accentRim="rgba(255,255,255,0.22)"
                    label="01 · DASHCAM PIPELINE"
                    title="Video Analysis"
                    description="Upload MP4/AVI dashcam footage. Run frame-by-frame pothole classification with configurable confidence thresholds."
                    cta="Launch Pipeline"
                    onClick={() => setActivePage("video-analysis")}
                    primary
                  />
                  <CTACard
                    icon={<Camera className="w-4 h-4" />}
                    accent="#E5E5E5"
                    accentDim="rgba(255,255,255,0.06)"
                    accentRim="rgba(255,255,255,0.22)"
                    label="02 · INSTANT TEST"
                    title="Image Classifier"
                    description="Drop a single road surface image for immediate InceptionV3 hazard classification and probability score."
                    cta="Test Image"
                    onClick={() => setActivePage("image-test")}
                  />
                  <CTACard
                    icon={<BarChart3 className="w-4 h-4" />}
                    accent="#D4D4D4"
                    accentDim="rgba(255,255,255,0.06)"
                    accentRim="rgba(255,255,255,0.22)"
                    label="03 · EVALUATION"
                    title="Model Metrics"
                    description="Inspect held-out test set accuracy, precision/recall curves, confusion matrix, and training convergence."
                    cta="View Metrics"
                    onClick={() => setActivePage("evaluation")}
                  />
                </div>
              </div>

              {/* Recent analysis */}
              {videoResults && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="section-label">Latest Analysis Results</p>
                    <button
                      onClick={() => setActivePage("video-analysis")}
                      className="font-mono text-[11px] flex items-center gap-1"
                      style={{ color: "var(--amber)" }}
                    >
                      Full frame timeline <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                  <PredictionResults results={videoResults} onReset={handleResetVideo} />
                </div>
              )}
            </div>
          )}

          {/* ── VIDEO ANALYSIS ───────────────────────────────────── */}
          {activePage === "video-analysis" && (
            <div className="space-y-6">
              {!uploadedVideo ? (
                <VideoUploader
                  onUploadSuccess={handleUploadSuccess}
                  isUploading={isUploading}
                  setIsUploading={setIsUploading}
                />
              ) : !videoResults ? (
                <div className="space-y-5">
                  <VideoPreview
                    videoUpload={uploadedVideo}
                    videoFile={uploadedVideoFile}
                    onStartProcessing={handleStartProcessing}
                    isProcessing={isProcessing}
                    onReset={handleResetVideo}
                  />
                  {processingStatus && <ProcessingProgress status={processingStatus} />}
                </div>
              ) : (
                <PredictionResults results={videoResults} onReset={handleResetVideo} />
              )}
            </div>
          )}

          {/* ── IMAGE TEST ───────────────────────────────────────── */}
          {activePage === "image-test" && <ImageTester />}

          {/* ── EVALUATION ──────────────────────────────────────── */}
          {activePage === "evaluation" && (
            <MetricsCharts metrics={evaluationMetrics} isLoading={isLoadingStatus} />
          )}

          {/* ── ABOUT ───────────────────────────────────────────── */}
          {activePage === "about" && <AboutPage />}
        </main>
      </div>
    </div>
  );
}

/* ── CTA Card ──────────────────────────────────────────────────────── */
interface CTACardProps {
  icon: React.ReactNode;
  accent: string;
  accentDim: string;
  accentRim: string;
  label: string;
  title: string;
  description: string;
  cta: string;
  onClick: () => void;
  primary?: boolean;
}

function CTACard({ icon, accent, accentDim, accentRim, label, title, description, cta, onClick, primary }: CTACardProps) {
  return (
    <div
      className="p-5 flex flex-col justify-between gap-4 cursor-pointer transition-all duration-150"
      style={{
        background: "var(--card)",
        border: "1px solid var(--rim)",
        borderTop: `2px solid ${accent}`,
        borderRadius: 4,
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.background = accentDim;
        (e.currentTarget as HTMLElement).style.borderColor = accentRim;
        (e.currentTarget as HTMLElement).style.borderTopColor = accent;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.background = "var(--card)";
        (e.currentTarget as HTMLElement).style.borderColor = "var(--rim)";
        (e.currentTarget as HTMLElement).style.borderTopColor = accent;
      }}
    >
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 flex items-center justify-center"
            style={{
              background: accentDim,
              border: `1px solid ${accentRim}`,
              borderRadius: 3,
              color: accent,
            }}
          >
            {icon}
          </div>
          <span className="font-mono text-[10px] font-bold tracking-widest uppercase" style={{ color: accent }}>
            {label}
          </span>
        </div>
        <div className="text-ink font-bold text-sm">{title}</div>
        <p className="font-mono text-[11px] leading-relaxed" style={{ color: "var(--muted)" }}>
          {description}
        </p>
      </div>

      <button
        onClick={onClick}
        className="btn"
        style={
          primary
            ? { background: accent, color: "var(--void)", borderColor: accent, fontSize: 12 }
            : { background: accentDim, color: accent, borderColor: accentRim, fontSize: 12 }
        }
      >
        {cta} <ArrowRight className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

/* ── About Page ────────────────────────────────────────────────────── */
function AboutPage() {
  return (
    <div className="space-y-5 animate-fade-up">
      {/* Hero */}
      <div
        className="p-6"
        style={{
          background: "var(--card)",
          border: "1px solid var(--rim)",
          borderLeft: "3px solid var(--amber)",
          borderRadius: 4,
        }}
      >
        <div className="flex items-start gap-4">
          <div
            className="p-2.5 shrink-0"
            style={{ background: "var(--amber-glow)", border: "1px solid var(--amber-rim)", borderRadius: 3 }}
          >
            <BookOpen className="w-5 h-5" style={{ color: "var(--amber)" }} />
          </div>
          <div className="space-y-2">
            <h2 className="text-lg font-bold text-ink leading-tight">
              Road Pothole Detection from Dashcam Footage Using Deep CNN
            </h2>
            <p className="font-mono text-[12px] leading-relaxed" style={{ color: "var(--muted)" }}>
              Automated road pavement distress inspection designed for edge vehicular dashcam deployment.
              Traditional municipal road surveys are labor-intensive and infrequent. By leveraging Transfer
              Learning with InceptionV3, this project achieves high-throughput binary hazard classification
              with low computational overhead.
            </p>
          </div>
        </div>
      </div>

      {/* Classification clarification */}
      <div
        className="p-4 flex gap-3"
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid var(--rim)",
          borderRadius: 4,
        }}
      >
        <Info className="w-4 h-4 shrink-0 mt-px text-white" />
        <div className="space-y-1">
          <div className="text-sm font-bold text-ink">Classification vs. Object Localization</div>
          <p className="font-mono text-[11px] leading-relaxed" style={{ color: "var(--muted)" }}>
            This system performs <strong className="text-ink">binary frame classification</strong> (Pothole vs Normal Road)
            — not pixel-level bounding boxes like YOLO/Mask R-CNN. Frame-level classification determines whether a
            vehicle is traversing over a distressed surface, providing GPS/timestamp-referenced telemetry for
            municipal maintenance prioritization.
          </p>
        </div>
      </div>

      {/* Architecture grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ArchBlock
          icon={<Cpu className="w-4 h-4 text-white" />}
          title="InceptionV3 Deep CNN Backbone"
          items={[
            "Pretrained on 1.28M ImageNet images across 1,000 categories.",
            "Multi-scale factorized convolutions (1×1, 3×3, 1×7, 7×1) for micro-texture and macro-feature extraction simultaneously.",
            "21.8M total parameters · Global Average Pooling prevents overfitting vs. dense flatten layers.",
            "Two-phase training: Phase 1 freezes backbone; Phase 2 fine-tunes upper blocks at 1e-5 LR.",
          ]}
        />
        <ArchBlock
          icon={<Layers className="w-4 h-4 text-white" />}
          title="Data Preprocessing & Integrity"
          items={[
            "Input normalized to 299×299×3, pixels scaled to [-1.0, 1.0] using InceptionV3 preprocess_input.",
            "SHA-256 duplicate hashing eliminates cross-split contamination between train/val/test sets.",
            "Stratified partitioning: 70% Train / 15% Validation / 15% Test (reproducible seed).",
            "Training-only augmentation: horizontal flip, ±4% rotation, ±5% zoom — preserves dashcam physics.",
          ]}
        />
      </div>

      {/* References */}
      <div
        className="p-5"
        style={{ background: "var(--card)", border: "1px solid var(--rim)", borderRadius: 4 }}
      >
        <p className="section-label mb-4">Key Academic References</p>
        <ol className="space-y-2 list-decimal list-inside">
          {[
            `Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." IEEE CVPR, pp. 2818–2826.`,
            `Kumar, A. (2020). "Pothole Detection Dataset." Kaggle. kaggle.com/datasets/atulyakumar98/pothole-detection-dataset.`,
            `Deng, J., et al. (2009). "ImageNet: A Large-Scale Hierarchical Image Database." IEEE CVPR.`,
          ].map((ref, i) => (
            <li key={i} className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              {ref}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function ArchBlock({ icon, title, items }: { icon: React.ReactNode; title: string; items: string[] }) {
  return (
    <div
      className="p-5 space-y-4"
      style={{ background: "var(--card)", border: "1px solid var(--rim)", borderRadius: 4 }}
    >
      <div className="flex items-center gap-2.5">
        {icon}
        <span className="text-sm font-bold text-ink">{title}</span>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2">
            <span
              className="font-mono text-[10px] font-bold mt-0.5 shrink-0"
              style={{ color: "var(--dim)" }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="font-mono text-[11px] leading-relaxed" style={{ color: "var(--muted)" }}>
              {item}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
