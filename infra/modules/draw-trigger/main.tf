terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# Starts the collector's GitHub workflows on time. GitHub's own `schedule`
# trigger has run 2-11 hours late on this repo since late August 2026 and
# drops runs outright, while a workflow_dispatch through the REST API starts
# within seconds. So AWS keeps the clock and GitHub still does the work: one
# authenticated POST per run, and nothing here executes code.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.region

  # A list, not a map: the API returns the headers in the order they were
  # sent, and the token command in outputs.tf has to send the same list.
  github_headers = [
    { key = "Accept", value = "application/vnd.github+json" },
    { key = "X-GitHub-Api-Version", value = "2022-11-28" },
  ]
  # No User-Agent, although GitHub requires one. EventBridge strips any
  # User-Agent it is given and always sends its own,
  # "Amazon/EventBridge/ApiDestinations", which satisfies GitHub.

  # Deliberately a well-formed header value that GitHub rejects with 401.
  # The real token is set out of band (see outputs.tf), so it never reaches
  # this file, a tfvars file, the state, or a plan. If nobody sets it, the
  # first scheduled run fails loudly through the alarms below.
  token_placeholder = "Bearer unset-see-infra-README"

  # Source for the hand-fired test event. Custom sources cannot start with
  # "aws.", so nothing else on the default bus can match this by accident.
  test_event_source = "${var.name_prefix}.draw-trigger"

  schedules = {
    collect-evening = {
      workflow    = "collect.yml"
      expression  = var.collect_evening_schedule
      description = "Dispatch collect.yml after the Wed/Sat draw"
    }
    collect-retry = {
      workflow    = "collect.yml"
      expression  = var.collect_retry_schedule
      description = "Dispatch collect.yml again the next morning"
    }
    watchdog = {
      workflow    = "watchdog.yml"
      expression  = var.watchdog_schedule
      description = "Dispatch watchdog.yml to check the draw landed"
    }
  }

  # Every rule that dispatches, scheduled or not. The test rule shares the
  # connection, the destination, the role and the dead-letter queue with the
  # real ones, which is the point: firing it proves the path the schedules use.
  rules = merge(
    { for key, rule in local.schedules : key => merge(rule, { pattern = null }) },
    {
      dispatch-test = {
        workflow    = "watchdog.yml"
        expression  = null
        description = "Dispatch watchdog.yml on a hand-sent test event"
        pattern     = jsonencode({ source = [local.test_event_source] })
      }
    },
  )

  rule_arns = [for rule in aws_cloudwatch_event_rule.dispatch : rule.arn]
}

# --- the call ----------------------------------------------------------------

resource "aws_cloudwatch_event_connection" "github" {
  name               = "${var.name_prefix}-github"
  description        = "GitHub REST API for ${var.github_repo}. Token is set by hand - see infra/README.md."
  authorization_type = "API_KEY"

  auth_parameters {
    # API_KEY auth sends `<key>: <value>` on every request, which is exactly
    # GitHub's `Authorization: Bearer <token>`.
    api_key {
      key   = "Authorization"
      value = local.token_placeholder
    }

    # Nested inside auth_parameters in this provider, which is why the
    # ignore_changes below names the token's own path rather than the whole
    # block: ignoring auth_parameters would stop managing these as well.
    invocation_http_parameters {
      dynamic "header" {
        for_each = local.github_headers
        content {
          key             = header.value.key
          value           = header.value.value
          is_value_secret = false
        }
      }
    }
  }

  lifecycle {
    # DescribeConnection never returns the key, so the provider reads it back
    # from state and the placeholder stays there for good. But an in-place
    # update of this resource re-sends the whole auth block, placeholder
    # included (resourceConnectionUpdate in the provider's
    # internal/service/events/connection.go). After any apply that changes
    # this connection, run the token command again.
    ignore_changes = [auth_parameters[0].api_key[0].value]
  }
}

# One destination for both workflows, with the file name as a path wildcard
# filled in by each target. A second destination would repeat the same
# connection, method and rate limit to differ in one path segment, and it
# would grant nothing new: the token can dispatch any workflow in the repo
# either way.
resource "aws_cloudwatch_event_api_destination" "dispatch" {
  name                = "${var.name_prefix}-workflow-dispatch"
  description         = "POST a workflow_dispatch for ${var.github_repo}"
  invocation_endpoint = "https://api.github.com/repos/${var.github_repo}/actions/workflows/*/dispatches"
  http_method         = "POST"
  connection_arn      = aws_cloudwatch_event_connection.github.arn

  # The lowest the API accepts. At most one call is ever due at a time.
  invocation_rate_limit_per_second = 1
}

# --- when --------------------------------------------------------------------

# Rules on the default bus rather than EventBridge Scheduler, whose timezone
# support would be nicer: Scheduler's targets are AWS API operations, and an
# HTTPS endpoint behind a connection is only reachable as a rule target.
resource "aws_cloudwatch_event_rule" "dispatch" {
  for_each = local.rules

  name                = "${var.name_prefix}-${each.key}"
  description         = each.value.description
  schedule_expression = each.value.expression
  event_pattern       = each.value.pattern
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "dispatch" {
  for_each = local.rules

  rule      = aws_cloudwatch_event_rule.dispatch[each.key].name
  target_id = "github-workflow-dispatch"
  arn       = aws_cloudwatch_event_api_destination.dispatch.arn
  role_arn  = aws_iam_role.invoke.arn

  # The request body. `inputs` is left out, so collect.yml's test_email falls
  # back to its default of false and a scheduled run never sends a test mail.
  input = jsonencode({ ref = var.git_ref })

  http_target {
    path_parameter_values = [each.value.workflow]
  }

  # EventBridge retries 401, 407, 409, 429 and 5xx and gives up at once on any
  # other 4xx. An hour covers a GitHub incident without keeping an expired
  # token's failure quiet for the default 24 hours: the alarm below cannot
  # fire until the retries are spent.
  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 5
  }

  dead_letter_config {
    arn = aws_sqs_queue.dlq.arn
  }
}

# --- identity ----------------------------------------------------------------

data "aws_iam_policy_document" "invoke_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    # Confused-deputy guard, as EventBridge documents it for rule targets:
    # only these rules, in this account, can hand this role to the service.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = local.rule_arns
    }
  }
}

resource "aws_iam_role" "invoke" {
  name                 = "${var.name_prefix}-draw-trigger"
  description          = "Lets the draw schedules call one API destination. Nothing else."
  assume_role_policy   = data.aws_iam_policy_document.invoke_trust.json
  max_session_duration = 3600
}

# One action on one destination. The role cannot reach any other destination
# in the account, and it never sees the token: EventBridge reads that from the
# connection's secret itself.
data "aws_iam_policy_document" "invoke" {
  statement {
    sid       = "InvokeTheDispatchDestination"
    effect    = "Allow"
    actions   = ["events:InvokeApiDestination"]
    resources = [aws_cloudwatch_event_api_destination.dispatch.arn]
  }
}

resource "aws_iam_role_policy" "invoke" {
  name   = "invoke-dispatch"
  role   = aws_iam_role.invoke.id
  policy = data.aws_iam_policy_document.invoke.json
}

# --- failure -----------------------------------------------------------------

# Where a dispatch lands once EventBridge has given up on it, with the rule,
# the error code and the retry count as message attributes. Kept for the
# 14-day maximum so a failure is still readable after a week away.
resource "aws_sqs_queue" "dlq" {
  name                      = "${var.name_prefix}-dispatch-dlq"
  message_retention_seconds = 1209600

  # SSE-SQS, not KMS. EventBridge cannot write to a queue encrypted with the
  # AWS-managed aws/sqs key, and the messages hold an event and an error, not
  # the token.
  sqs_managed_sse_enabled = true
}

# EventBridge attaches this itself only when a target is created in the
# console. Through the API it has to exist, or dead letters are dropped.
data "aws_iam_policy_document" "dlq" {
  statement {
    sid       = "DeadLettersFromTheDispatchRules"
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.dlq.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = local.rule_arns
    }
  }
}

resource "aws_sqs_queue_policy" "dlq" {
  queue_url = aws_sqs_queue.dlq.id
  policy    = data.aws_iam_policy_document.dlq.json
}

# No KMS on the topic for the same reason as the queue: CloudWatch cannot
# publish to a topic under the AWS-managed aws/sns key.
resource "aws_sns_topic" "alarms" {
  name = "${var.name_prefix}-dispatch-alarms"
}

# The statement CloudWatch documents for alarm notifications, in place of the
# default topic policy, which only works through the deprecated
# aws:SourceOwner key. Management of the topic still goes through IAM, which
# is enough inside one account.
data "aws_iam_policy_document" "alarms" {
  statement {
    sid       = "AlarmsFromThisModule"
    effect    = "Allow"
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.alarms.arn]

    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudwatch:${local.region}:${local.account_id}:alarm:${var.name_prefix}-*"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "alarms" {
  arn    = aws_sns_topic.alarms.arn
  policy = data.aws_iam_policy_document.alarms.json
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# An expired or revoked token is the failure this is built for. GitHub
# answers 401, which EventBridge retries until the policy above runs out and
# then sends to the dead-letter queue - so the queue alarm is the one that
# is guaranteed to see it. A 403 (token lacks actions:write) or 404 (wrong
# repo or file) skips the retries and goes straight there.
#
# Both alarms treat missing data as fine: between draws there are no
# invocations at all, and EventBridge publishes FailedInvocations only when
# it is non-zero.
resource "aws_cloudwatch_metric_alarm" "dispatch_failed" {
  for_each = local.schedules

  alarm_name        = "${var.name_prefix}-${each.key}-dispatch-failed"
  alarm_description = "${aws_cloudwatch_event_rule.dispatch[each.key].name} could not dispatch ${each.value.workflow}. Most likely the GitHub token expired: see infra/README.md."

  namespace   = "AWS/Events"
  metric_name = "FailedInvocations"
  # RuleName alone is how EventBridge labels rules on the default bus.
  dimensions = {
    RuleName = aws_cloudwatch_event_rule.dispatch[each.key].name
  }

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}

# Covers every rule, the hand-fired test included, which is why the test rule
# has no alarm of its own. The mail is the signal, not the alarm state: SQS
# stops publishing metrics for a queue six hours after it was last touched,
# so this drops back to OK on its own with the message still inside, and the
# next dead letter wakes the queue and alarms again (up to 15 minutes late,
# per the SQS docs on inactive queues).
resource "aws_cloudwatch_metric_alarm" "dlq_not_empty" {
  alarm_name        = "${var.name_prefix}-dispatch-dlq-not-empty"
  alarm_description = "A GitHub workflow dispatch was given up on. The message attributes in ${aws_sqs_queue.dlq.name} say which rule and why."

  namespace   = "AWS/SQS"
  metric_name = "ApproximateNumberOfMessagesVisible"
  dimensions = {
    QueueName = aws_sqs_queue.dlq.name
  }

  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
}
