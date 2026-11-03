"use client";

import React from "react";
import {
  LayoutDashboard,
  Video,
  Camera,
  BarChart3,
  BookOpen,
  Activity,
  Cpu,
} from "lucide-react";
import { NavPage, ModelStatus } from "../lib/types";

interface SidebarProps {
  activePage: NavPage;
  setActivePage: (page: NavPage) => void;
  modelStatus: ModelStatus | null;
  backendOnline: boolean;
}

const navItems: { id: NavPage; label: string; icon: React.ElementType; sub: string }[] = [
  { id: "dashboard",      label: "Dashboard",      icon: LayoutDashboard, sub: "Overview & telemetry" },
  { id: "video-analysis", label: "Video Analysis",  icon: Video,           sub: "Frame inference pipeline" },
  { id: "image-test",     label: "Image Test",      icon: Camera,          sub: "Single frame classifier" },
  { id: "evaluation",     label: "Evaluation",      icon: BarChart3,       sub: "Metrics & curves" },
  { id: "about",          label: "About",           icon: BookOpen,        sub: "Architecture & methods" },
];

export default function Sidebar({
  activePage,
  setActivePage,
  modelStatus,
  backendOnline,
}: SidebarProps) {
  return (
    <aside
      className="w-56 flex flex-col justify-between shrink-0 h-screen sticky top-0"
      style={{
        background: "var(--surface)",
        borderRight: "1px solid var(--rim)",
      }}
    >
      <div>
        {/* ── Navigation ────────────────────────────────── */}
        <nav className="p-2 pt-4 space-y-0.5">
          <p className="section-label px-2 mb-2">Navigation</p>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`nav-item ${isActive ? "active" : ""}`}
              >
                <Icon
                  className="nav-icon w-4 h-4 shrink-0"
                  style={{ color: isActive ? "var(--amber)" : "var(--dim)" }}
                />
                <div className="flex-1 text-left">
                  <div style={{ fontSize: 13 }}>{item.label}</div>
                  <div
                    className="font-mono text-[10px] mt-px"
                    style={{ color: isActive ? "var(--amber-light)" : "var(--dim)", opacity: 0.9 }}
                  >
                    {item.sub}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* ── System status footer ──────────────────────── */}
      <div
        className="p-3 space-y-2"
        style={{ borderTop: "1px solid var(--rim)", background: "var(--raised)" }}
      >
        <p className="section-label mb-2">System</p>

        <div className="flex items-center justify-between">
          <span className="flex items-center gap-2 font-mono text-[11px]" style={{ color: "var(--muted)" }}>
            <Activity className="w-3 h-3" />
            API Server
          </span>
          <span
            className="status-pill"
            style={{
              background: backendOnline ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)",
              color: backendOnline ? "#FFFFFF" : "var(--dim)",
              border: `1px solid ${backendOnline ? "rgba(255,255,255,0.22)" : "var(--rim)"}`,
            }}
          >
            <span
              className={backendOnline ? "animate-pulse-dot" : ""}
              style={{
                width: 6, height: 6, borderRadius: "50%",
                background: backendOnline ? "#FFFFFF" : "var(--dim)",
                display: "inline-block",
              }}
            />
            {backendOnline ? "Live" : "Down"}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="flex items-center gap-2 font-mono text-[11px]" style={{ color: "var(--muted)" }}>
            <Cpu className="w-3 h-3" />
            InceptionV3
          </span>
          <span
            className="status-pill"
            style={{
              background: modelStatus?.is_loaded
                ? "rgba(255,255,255,0.08)"
                : "rgba(255,255,255,0.03)",
              color: modelStatus?.is_loaded
                ? "#FFFFFF"
                : "var(--dim)",
              border: `1px solid ${
                modelStatus?.is_loaded
                  ? "rgba(255,255,255,0.22)"
                  : "var(--rim)"
              }`,
            }}
          >
            <span
              style={{
                width: 6, height: 6, borderRadius: "50%",
                background: modelStatus?.is_loaded ? "#FFFFFF" : "var(--dim)",
                display: "inline-block",
              }}
            />
            {modelStatus?.is_loaded ? "Ready" : modelStatus?.is_available ? "Idle" : "None"}
          </span>
        </div>
      </div>
    </aside>
  );
}
