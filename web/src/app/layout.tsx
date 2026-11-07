import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { Suspense } from "react";
import { HealthPill, Nav } from "@/components/Nav";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ClauseCompass",
  description: "Contract QA and compliance auditing where every citation is verified in code.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans md:flex-row">
        <aside className="border-b border-line bg-surface md:sticky md:top-0 md:h-screen md:w-60 md:shrink-0 md:border-r md:border-b-0">
          <div className="flex h-full flex-col gap-4 p-3 md:p-4">
            <Link href="/" className="flex items-center gap-2 px-3 pt-1">
              <span className="grid size-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white dark:text-bg">
                §
              </span>
              <span className="font-semibold tracking-tight">ClauseCompass</span>
            </Link>
            <Suspense>
              <Nav />
            </Suspense>
            <div className="mt-auto hidden border-t border-line pt-3 md:block">
              <HealthPill />
              <p className="px-3 text-xs leading-relaxed text-ink-3">
                Every citation is checked against the retrieved source before it is shown.
              </p>
            </div>
          </div>
        </aside>
        <main className="min-w-0 flex-1 px-4 py-6 md:px-10 md:py-10">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </body>
    </html>
  );
}
