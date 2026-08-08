/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: the whole site is objects in S3 behind CloudFront, with no
  // server to render on. Everything below follows from that.
  output: 'export',

  // Emit out/foo/index.html rather than out/foo.html. The S3 REST endpoint
  // (which is what a private bucket behind Origin Access Control exposes) does
  // not resolve index documents, so a CloudFront Function rewrites the URL -
  // see infra/modules/static-site/functions/rewrite.js. Directory-style output
  // is what that rewrite expects.
  trailingSlash: true,

  // next/image needs a server to optimise on. There isn't one.
  images: { unoptimized: true },

  // A broken type must fail the build, not ship. Linting is a separate step
  // (`npm run lint`) since Next 16 dropped the config key for it.
  typescript: { ignoreBuildErrors: false },

  // Next writes its own AGENTS.md/CLAUDE.md on build. This repo documents its
  // conventions at the root, and a generated pair under site/ is noise that
  // reappears on every build.
  agentRules: false,
};

export default nextConfig;
