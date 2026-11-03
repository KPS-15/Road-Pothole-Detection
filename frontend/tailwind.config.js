/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ['Space Grotesk', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        void:    "#080A0F",
        surface: "#0D1117",
        raised:  "#121820",
        card:    "#161E28",
        panel:   "#1A2332",
        rim:     "#1F2D3D",
        rim2:    "#263548",

        amber: {
          DEFAULT: "#F97316",
          light:   "#FB923C",
          dark:    "#EA6C0D",
          glow:    "rgba(249,115,22,0.15)",
        },
        acid: {
          DEFAULT: "#A3E635",
          dim:     "rgba(163,230,53,0.12)",
        },
        hazard: {
          DEFAULT: "#EF4444",
          dim:     "rgba(239,68,68,0.12)",
        },
        ice: {
          DEFAULT: "#38BDF8",
          dim:     "rgba(56,189,248,0.10)",
        },
        ink:  "#F0F6FF",
        muted: "#8B9EB5",
        dim:   "#4A6070",
      },
      borderRadius: {
        sm: "2px",
        DEFAULT: "3px",
        md: "4px",
        lg: "6px",
      },
      boxShadow: {
        amber: "0 4px 24px -8px rgba(249,115,22,0.35)",
        "amber-sm": "0 0 0 1px rgba(249,115,22,0.25), 4px 0 24px -8px rgba(249,115,22,0.15)",
        hazard: "0 4px 24px -8px rgba(239,68,68,0.35)",
        acid:   "0 4px 24px -8px rgba(163,230,53,0.25)",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.3" },
        },
        "scan-line": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(200%)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
        "scan-line": "scan-line 2.5s linear infinite",
        "fade-up":   "fade-up 0.3s ease-out",
      },
    },
  },
  plugins: [],
};
