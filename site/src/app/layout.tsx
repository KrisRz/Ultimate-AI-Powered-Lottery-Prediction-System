import type { Metadata, Viewport } from 'next';
import { Archivo, JetBrains_Mono, Literata } from 'next/font/google';

import './globals.css';

/**
 * Three faces, three jobs.
 *
 * Archivo carries the width axis, which is the point: set expanded and tight,
 * it reads like the officialdom printed on a betting slip without becoming a
 * pastiche of one. Literata is the reading face - a sturdy screen serif rather
 * than one of the high-contrast display serifs every explainer reaches for.
 * JetBrains Mono sets every figure on the page, and this page is mostly
 * figures.
 *
 * All three are self-hosted at build time by next/font, so the strict CSP in
 * front of the CDN never has to allow a third-party font host.
 */
const display = Archivo({
  subsets: ['latin'],
  axes: ['wdth'],
  variable: '--font-display',
  display: 'swap',
});

const body = Literata({
  subsets: ['latin'],
  variable: '--font-body',
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '700'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'The Lotto EV Toolkit — honest arithmetic about a game you cannot beat',
  description:
    'A walk through an expected-value model for UK Lotto: why no method predicts the numbers, when a ticket is briefly worth more than it costs, and how the whole thing is collected and operated.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  colorScheme: 'light dark',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-GB" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#summary">
          Skip to the plain-text summary
        </a>
        {children}
      </body>
    </html>
  );
}
