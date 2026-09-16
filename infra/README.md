# Infrastructure

Hosting for the public explainer at `lotto.krisgrzepka.com`, and the clock
that starts the draw collector. Static files on a private S3 bucket, served
through CloudFront, deployed from GitHub Actions with no long-lived
credentials. EventBridge rules that ask GitHub to run the collection
workflows on time.

**The lottery toolkit still does not run here.** Collection runs on GitHub
Actions, settlement stays on the laptop's launchd job, and the data stays in
Git. What moved is the trigger: AWS now decides *when* `collect.yml` and
`watchdog.yml` run and starts them through GitHub's workflow-dispatch API,
because GitHub's own cron had been starting them 2–11 hours late and dropping
runs. There is still no compute in this stack. No Lambda and no container,
just a scheduled HTTPS request.

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

```
   EventBridge rules (UTC)         API destination                   GitHub
   Wed/Sat 21:50  collect   ─┐     POST /repos/…/actions/        ┌──────────────┐
   Thu/Sun 06:05  collect   ─┼───▶ workflows/{file}/dispatches ─▶│ collect.yml  │
   Thu/Sun 12:05  watchdog  ─┘     token from the connection     │ watchdog.yml │
                                                                 └──────────────┘
          │ retries spent
          ▼
   SQS dead-letter queue ──▶ CloudWatch alarms ──▶ SNS ──▶ email
```

## Layout

| Path | What it is |
|---|---|
| `bootstrap/` | The state bucket. Run once, keeps its own state on disk. |
| `live/` | The stack: zone lookup, certificate, site module, DNS, deploy role, draw trigger, budget. |
| `modules/static-site/` | Bucket, Origin Access Control, distribution, headers, error mapping. |
| `modules/github-oidc/` | The federated provider and the one role CI is allowed to assume. |
| `modules/draw-trigger/` | GitHub connection, API destination, the schedules and their role, dead-letter queue, failure alarms. |

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

## The draw trigger

Added to an existing stack, it goes in on its own, like every other change:

```bash
cd infra/live
terraform init                                  # the module is new; no -upgrade, the lock file stays put
terraform plan  -target=module.draw_trigger     # 21 to add, 0 to change, 0 to destroy
terraform apply -target=module.draw_trigger
```

**Then give it a token.** The dispatch call authenticates with a fine-grained
GitHub token, and Terraform never sees it. On GitHub: Settings → Developer
settings → Fine-grained tokens. Give it access to this one repository only,
with the repository permission **Actions: Read and write** (GitHub adds
Metadata: read by itself). Note the expiry date. An expired token is the
failure the alarms below are for. Then, from `infra/live`:

```bash
bash -c "$(terraform output -raw set_token_command)"    # hidden prompt; prints AUTHORIZED
```

The command contains no secret. It prompts for the token, builds the request
in a 0600 temp file that it deletes on exit, and calls
`aws events update-connection`. Rotating the token is the same command with the
new value, and Terraform does not change.

⚠️ **Run it again after any apply whose plan shows
`~ module.draw_trigger.aws_cloudwatch_event_connection.github`.** An in-place
update of the connection re-sends the whole auth block, and the placeholder
goes with it.

**Confirm the subscription.** AWS mails a confirmation link to
`budget_alert_email`. Alarms go nowhere until that link is clicked.

**Prove the path end to end.** Do this while the last draw is already in the
data. The test dispatches `watchdog.yml`, which mails if the draw is missing:

```bash
bash -c "$(terraform output -raw test_dispatch_command)"   # prints 0: event accepted
gh run list --workflow watchdog.yml --limit 1              # a workflow_dispatch run, seconds old
```

If no run appears, the dead-letter queue says why:

```bash
aws sqs receive-message --queue-url "$(terraform output -raw draw_trigger_dead_letter_queue_url)" \
  --message-attribute-names All --query 'Messages[].MessageAttributes'
aws sqs purge-queue --queue-url "$(terraform output -raw draw_trigger_dead_letter_queue_url)"   # once fixed
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

**AWS keeps the clock and GitHub still does the work.** GitHub's `schedule`
trigger ran the collector 2–11 hours late from late August 2026 and dropped
runs. A `workflow_dispatch` through the REST API starts within seconds. The
crons stay in the workflows as a fallback. Ingestion is idempotent, so
whichever run of a pair comes second adds nothing. The rules sit on the
default bus rather than in EventBridge Scheduler. Scheduler would take a
timezone, but its targets are AWS API operations, and it cannot call an HTTPS
endpoint. All times are UTC, like GitHub's. The draw at about 20:00 UK time is
19:00 or 20:00 UTC, and 21:50 is after both.

**The token is never in Terraform.** The connection is created with a
placeholder, `Bearer unset-see-infra-README`, which GitHub rejects. If nobody
sets the real token, the first scheduled run alarms. `ignore_changes` names
exactly `auth_parameters[0].api_key[0].value`, not the whole
`auth_parameters`, because in this provider that block also holds the
headers, and those should stay managed. `DescribeConnection` never returns the
key, so state keeps the placeholder for good. The catch is the one in the
warning above: the provider's update re-sends the whole block
(`resourceConnectionUpdate` in `internal/service/events/connection.go`). For
the same reason, the token command sends the complete request, key and
headers. How `UpdateConnection` treats a partial block is not documented.

**No `User-Agent` header.** GitHub requires one, but EventBridge
[removes any `User-Agent` it is given](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-api-destinations.html#eb-api-destination-headers)
and sends its own, `Amazon/EventBridge/ApiDestinations`, which GitHub accepts.

**One API destination, and the workflow file is a path wildcard.**
`…/actions/workflows/*/dispatches` is filled in by each target's
`path_parameter_values`. Two destinations would differ in one path segment
and grant nothing extra, because the token can dispatch any workflow in the
repository either way.

**What an expired token looks like.** GitHub answers 401. EventBridge
[retries 401, 407, 409, 429 and 5xx, and drops any other 4xx without a retry](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-api-destinations.html#eb-api-destination-error-codes).
After the last retry the event goes to the dead-letter queue (same page).
The retry policy is capped at one hour and five attempts, so the failure
surfaces the same evening rather than after the 24-hour default. Then
`lotto-ev-dispatch-dlq-not-empty` fires. That is the alarm the documentation
guarantees for a 401. The per-rule `lotto-ev-*-dispatch-failed` alarms watch
`FailedInvocations`, which counts invocations that
[failed permanently](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-monitoring.html).
The [dead-letter page](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-rule-dlq.html)
calls the queue metrics *additional* to that count, so these alarms should
fire alongside the queue alarm. Nobody has watched that happen in this
account yet. A 403 (token without `actions: write`) or a 404 (wrong repository
or file name) skips the retries and alarms within minutes. SQS stops
publishing metrics six hours after a queue was last touched, so the queue
alarm returns to OK by itself. The email is the signal, not the alarm state.

## Cost

Route 53 charges $0.50/month for the hosted zone, which the portfolio domain
already pays. Alias queries to CloudFront are free, ACM certificates are free,
and origin fetches from S3 to CloudFront are not billed as transfer. What is
left is CloudFront egress at roughly $0.085/GB in Europe — about nine cents a
month at a thousand visits.

The draw trigger adds about **$0.40 a month**, and all of it is alarms. The
prices below are eu-west-2 on-demand, taken from the AWS Price List API
(`/offers/v1.0/aws/<service>/current/eu-west-2/index.json`, published
2026-09-11 to 09-15):

| Item | List price | Here |
|---|---|---|
| API destination invocations | $0.24 per million | about 26 calls a month (6 a week), so effectively $0 |
| Scheduled rule invocations | first 14 million free | $0 |
| The connection's Secrets Manager secret | [included in the API destination charge](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-target-connection-auth.html#eb-target-connection-auth-considerations), where a standalone secret would be $0.40/month | $0 |
| CloudWatch alarms | $0.10 per standard-resolution alarm a month. The 10 free ones are already taken: this account had 17 in eu-west-2 in September 2026 | 4 alarms, $0.40 |
| SQS | first 1 million requests a month free, then $0.40 per million | $0 |
| SNS email | first 1,000 a month free, then $2.00 per 100,000 | $0 |
| The hand-fired test event | $1.00 per million custom events | $0 |

The rules, queue, topic, alarms and role carry the `Project` tag through
`default_tags`. The connection, API destination and targets cannot carry it,
because the EventBridge API has no tags for them. Their per-call charge (a
fraction of a cent a year) is therefore outside the budget's filter. Whether
Cost Explorer attributes alarm charges by tag has not been checked.

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

The draw trigger has no automated check. CI only validates it. Its proof is
the test dispatch under [The draw trigger](#the-draw-trigger), and after
that, a `workflow_dispatch` run of `collect.yml` within a minute of 21:50 UTC
on a draw night.
