import {
  HealthStatus,
  ModelStatus,
  ImagePrediction,
  VideoUploadResult,
  VideoStatus,
  VideoResults,
  EvaluationMetrics,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/api/model/status`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Model status check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchEvaluationMetrics(): Promise<EvaluationMetrics> {
  const res = await fetch(`${API_BASE}/api/evaluation/metrics`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Evaluation metrics fetch failed: ${res.statusText}`);
  return res.json();
}

export async function predictSingleImage(
  file: File,
  threshold: number = 0.50
): Promise<ImagePrediction> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("threshold", threshold.toString());

  const res = await fetch(`${API_BASE}/api/predict/image`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Image prediction failed");
  }

  return res.json();
}

export async function uploadDashcamVideo(file: File): Promise<VideoUploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/videos`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Video upload failed");
  }

  return res.json();
}

export async function startVideoProcessing(
  jobId: string,
  threshold: number = 0.50,
  frameInterval: number = 3,
  generateAnnotated: boolean = true
): Promise<VideoStatus> {
  const res = await fetch(`${API_BASE}/api/videos/${jobId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      threshold,
      frame_interval: frameInterval,
      generate_annotated_video: generateAnnotated,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to start video processing");
  }

  return res.json();
}

export async function fetchVideoStatus(jobId: string): Promise<VideoStatus> {
  const res = await fetch(`${API_BASE}/api/videos/${jobId}/status`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Status query failed: ${res.statusText}`);
  return res.json();
}

export async function fetchVideoResults(jobId: string): Promise<VideoResults> {
  const res = await fetch(`${API_BASE}/api/videos/${jobId}/results`, { cache: "no-store" });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to retrieve results");
  }
  return res.json();
}

export function getVideoDownloadUrl(jobId: string): string {
  return `${API_BASE}/api/videos/${jobId}/download`;
}

export function getStaticArtifactUrl(relativePath: string): string {
  return `${API_BASE}/static/${relativePath}`;
}
