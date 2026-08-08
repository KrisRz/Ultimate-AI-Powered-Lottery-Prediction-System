# Infrastructure

Hosting for the public explainer at `lotto.krisgrzepka.com`. Static files on a
private S3 bucket, served through CloudFront, deployed from GitHub Actions with
no long-lived credentials.

**None of the lottery toolkit runs here.** Collection stays on GitHub Actions,
settlement stays on the laptop's launchd job, and the data stays in Git. AWS
serves files and nothing else — there is no compute in this stack at all.

```
                       ┌──────────────┐
   push to main ──────▶│  Actions     │  build, then assume a role via OIDC
                       │  site-deploy │
                       └──────┬───────┘
                              │ s3 sync + invalidate
                              ▼
   ┌─────────────┐     ┌──────────────┐     ┌────────────┐
   │  Route 53   │────▶│  CloudFront  │────▶│  S3 (OAC)  │  private, no ACLs
   │  A + AAAA   │     │  + function  │     │            │
   └─────────────┘     └──────────────┘     └────────────┘
                          ACM (us-east-1)
```

## Layout

| Path | What it is |
|---|---|
| `bootstrap/` | The state bucket. Run once, keeps its own state on disk. |
| `live/` | The stack: zone lookup, certificate, site module, DNS, deploy role, budget. |
| `modules/static-site/` | Bucket, Origin Access Control, distribution, headers, error mapping. |
| `modules/github-oidc/` | The federated provider and the one role CI is allowed to assume. |

## First run

```bash
cd infra/bootstrap
terraform init && terraform apply          # creates the state bucket

cd ../live
terraform init                              # now the S3 backend exists
terraform apply
```

The apply pauses at `aws_acm_certificate_validation` for anywhere from a few
minutes to half an hour. That is the certificate authority answering a DNS
challenge, not Terraform hanging.

Then publish the outputs as **repository variables** — they are not secrets, and
`vars` makes a fork fail loudly rather than deploy somewhere unexpected:

```bash
gh variable set SITE_BUCKET          --body "$(terraform output -raw bucket_name)"
gh variable set SITE_DISTRIBUTION_ID --body "$(terraform output -raw distribution_id)"
gh variable set SITE_DOMAIN          --body "$(terraform output -raw site_domain)"
gh variable set AWS_DEPLOY_ROLE_ARN  --body "$(terraform output -raw deploy_role_arn)"
```

## Decisions worth knowing about

**The bucket name has no dots.** Every other static site in this account is
named after its domain. This one cannot be: CloudFront reaches an
Origin-Access-Control bucket over HTTPS at `<bucket>.s3.<region>.amazonaws.com`,
and the wildcard certificate there matches a single label. A bucket called
`lotto.krisgrzepka.com` fails TLS on every origin fetch. A `validation` block on
the module variable rejects it rather than leaving the next person to find out.

**No S3 website configuration.** The website endpoint resolves index documents,
but it is HTTP-only and cannot be private. Using the REST endpoint instead means
a CloudFront Function has to do that job — `modules/static-site/functions/rewrite.js`,
about ten lines, running at the edge for roughly a sixth of what Lambda@Edge costs.

**403 and 404 both map to the 404 page.** S3 answers a missing key with 403,
not 404, because the policy grants `GetObject` and deliberately not
`ListBucket`. Mapping only 404 would surface every typo as Access Denied.
Mapping either to `200 /index.html` — the single-page-app reflex — would tell
search engines that every typo is a real page.

**The deploy role cannot change the infrastructure.** It can list one bucket,
write objects into it, and invalidate one distribution. It cannot touch the
bucket policy, the distribution, or anything else in the account, so a
compromised workflow can deface the page but not repoint it. There is
deliberately no broader CI role: applies happen from a laptop with a human
reading the diff.

**The OIDC trust policy pins the exact ref** with `StringEquals`, not a
`StringLike` wildcard. `repo:owner/name:*` would let any branch in the
repository deploy to production.

**`script-src` allows `'unsafe-inline'`.** Next's static export inlines a
bootstrap script and a nonce needs a server to generate it. Everything else in
the policy is locked to `'self'`; no third-party host is contacted, which is
also why the fonts are self-hosted at build time. This is a real compromise, not
a clean result.

**Nothing is pinned to `Z2FDTNDATAQYW2`.** CloudFront's hosted zone id is read
from the distribution. Hardcoding it works right up until it does not.

## Cost

Route 53 charges $0.50/month for the hosted zone, which the portfolio domain
already pays. Alias queries to CloudFront are free, ACM certificates are free,
and origin fetches from S3 to CloudFront are not billed as transfer. What is
left is CloudFront egress at roughly $0.085/GB in Europe — about nine cents a
month at a thousand visits.

`aws_budgets_budget` mails at 80% of a forecast $5/month, filtered to this
stack's `Project` tag. It is an alarm, not a cap: at this size, approaching it
means something is broken rather than popular.

## Verification

`site/scripts/smoke.sh` runs as the final deploy step, so a deploy that breaks
one of these goes red:

| Assertion | What it proves |
|---|---|
| `/` returns 200 with `max-age=0` | the HTML header pass ran |
| a hashed asset is `immutable` | the asset pass ran, and visitors are not re-downloading the bundle |
| a missing path returns 404 | the 403→404 mapping is in place |
| `http://` redirects | `redirect-to-https` on the behaviour |
| the bucket URL returns 403 | the origin is genuinely private |
| HSTS and CSP present | the response-headers policy is attached |
| `content-encoding: br` | compression is on |
