import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rice Leaf Disease Checker — E5",
  description:
    "Federated EfficientNetV2-S + DINOv2 classifier for 19 rice-leaf diseases. Final E5 model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
