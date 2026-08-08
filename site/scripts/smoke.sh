#!/usr/bin/env bash
#
# Post-deploy checks, run as the last step so a broken deploy goes red rather
# than sitting there looking green. Each assertion covers a specific way this
# particular stack fails - the cache headers landing on the wrong pass, the
# 403-vs-404 mapping, the bucket quietly becoming reachable on its own.
#
#   bash scripts/smoke.sh https://lotto.krisgrzepka.com lotto-ev-site-123 eu-west-2

set -euo pipefail

BASE="${1:?usage: smoke.sh <base-url> [bucket] [region]}"
BUCKET="${2:-}"
REGION="${3:-eu-west-2}"

pass=0
fail=0

check() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$actual" == *"$expected"* ]]; then
    printf '  ok    %s\n' "$label"
    pass=$((pass + 1))
  else
    printf '  FAIL  %s\n         expected: %s\n         actual:   %s\n' "$label" "$expected" "$actual"
    fail=$((fail + 1))
  fi
}

headers() { curl -sS -I --max-time 20 "$@" | tr -d '\r'; }

echo "Smoke testing ${BASE}"

home="$(headers "${BASE}/")"
check "home returns 200"            "200"                     "$(head -1 <<<"$home")"
check "home is revalidated"         "max-age=0"               "$(grep -i '^cache-control:' <<<"$home" || true)"
check "HSTS is set"                 "max-age=31536000"        "$(grep -i '^strict-transport-security:' <<<"$home" || true)"
check "CSP is set"                  "default-src 'none'"      "$(grep -i '^content-security-policy:' <<<"$home" || true)"
check "nosniff is set"              "nosniff"                 "$(grep -i '^x-content-type-options:' <<<"$home" || true)"

# Brotli proves compress=true on the behaviour; without it the page ships
# roughly four times the bytes.
check "brotli is negotiated" "br" \
  "$(curl -sS -I --max-time 20 -H 'Accept-Encoding: br' "${BASE}/" | tr -d '\r' | grep -i '^content-encoding:' || true)"

# A hashed asset must be immutable, or every visit re-downloads the bundle.
asset="$(curl -sS --max-time 20 "${BASE}/" | grep -o '/_next/static/[^"]*\.js' | head -1 || true)"
if [[ -n "$asset" ]]; then
  check "hashed asset is immutable" "immutable" \
    "$(headers "${BASE}${asset}" | grep -i '^cache-control:' || true)"
else
  echo "  FAIL  could not find a hashed asset in the homepage"
  fail=$((fail + 1))
fi

# S3 answers a missing key with 403 because the deploy role has no
# ListBucket. Both codes are mapped to the 404 page; if only one were, typos
# would surface as Access Denied.
check "missing page returns 404" "404" \
  "$(head -1 <<<"$(headers "${BASE}/definitely-not-a-real-page/")")"

check "http redirects to https" "301" \
  "$(head -1 <<<"$(headers "${BASE/https:/http:}/")")"

# The origin must not be reachable except through CloudFront.
if [[ -n "$BUCKET" ]]; then
  check "origin bucket is private" "403" \
    "$(head -1 <<<"$(headers "https://${BUCKET}.s3.${REGION}.amazonaws.com/index.html")")"
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ "$fail" -eq 0 ]]
