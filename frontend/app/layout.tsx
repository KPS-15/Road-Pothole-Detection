import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoadEye — Pothole Detection System",
  description: "Real-time road hazard classification from dashcam footage using InceptionV3 deep CNN transfer learning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-void text-ink min-h-screen antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
