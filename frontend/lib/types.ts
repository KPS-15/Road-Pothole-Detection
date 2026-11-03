export type NavPage = "dashboard" | "video-analysis" | "image-test" | "evaluation" | "about";

export interface HealthStatus {
  status: string;
  version: string;
  device: string;
  gpu_available: boolean;
}

export interface ModelStatus {
  is_available: boolean;
  is_loaded: boolean;
  model_path: string;
  architecture: string;
  input_shape: number[];
  total_params?: number;
  config?: {
    batch_size?: number;
    dropout_rate?: number;
    head_learning_rate?: number;
    fine_tune_learning_rate?: number;
    head_epochs?: number;
    fine_tune_epochs?: number;
    [key: string]: any;
  };
}

export interface ImagePrediction {
  filename: string;
  predicted_class: "Pothole" | "Normal";
  pothole_probability: number;
  confidence_percentage: number;
  is_hazard: boolean;
  inference_time_ms: number;
  threshold_applied: number;
}

export interface VideoUploadResult {
  job_id: string;
  original_filename: string;
  file_size_bytes: number;
  duration_sec: number;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  upload_time: string;
}

export interface VideoStatus {
  job_id: string;
  status: "uploaded" | "processing" | "completed" | "failed";
  progress_percentage: number;
  current_frame: number;
  total_frames: number;
  processed_frames: number;
  pothole_count: number;
  normal_count: number;
  error?: string;
}

export interface FramePrediction {
  frame_number: number;
  timestamp_sec: number;
  predicted_class: "Pothole" | "Normal";
  pothole_probability: number;
  is_pothole: boolean;
  inference_time_ms: number;
  thumbnail_data_url?: string;
}

export interface VideoResults {
  job_id: string;
  status: string;
  original_filename: string;
  total_video_frames: number;
  evaluated_frames_count: number;
  pothole_frames_count: number;
  normal_frames_count: number;
  pothole_percentage: number;
  average_latency_ms: number;
  estimated_fps: number;
  total_processing_time_sec: number;
  has_annotated_video: boolean;
  frame_timeline: FramePrediction[];
  top_pothole_frames: FramePrediction[];
}

export interface EvaluationMetrics {
  has_results: boolean;
  summary?: {
    overall_metrics?: {
      accuracy: number;
      precision_pothole: number;
      recall_pothole: number;
      f1_score_pothole: number;
      specificity_normal: number;
      roc_auc: number;
      pr_auc: number;
    };
    confusion_matrix?: {
      true_negatives_normal: number;
      false_positives_false_alert: number;
      false_negatives_missed_pothole: number;
      true_positives_detected_pothole: number;
    };
    latency_benchmark?: {
      mean_latency_per_image_ms: number;
      approx_throughput_fps: number;
    };
    test_dataset_size?: number;
    decision_threshold?: number;
  };
  classification_report?: string;
  dataset_manifest?: {
    total_scanned?: number;
    unique_valid?: number;
    duplicate_count?: number;
    splits?: {
      train?: { Normal: number; Pothole: number };
      validation?: { Normal: number; Pothole: number };
      test?: { Normal: number; Pothole: number };
    };
  };
}
