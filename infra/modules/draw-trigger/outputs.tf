# The token command sends the same request the provider sends on an update -
# name, auth type, key and the full header list - with only the key's value
# swapped for the real token. UpdateConnection's handling of a partial
# AuthParameters block is not documented, so the command never relies on it.
#
# The token is read with a hidden prompt, reaches jq through the environment
# rather than argv (process arguments are visible to `ps`), and sits in a
# 0600 temp file for the second the CLI needs: the AWS CLI rejects a pipe for
# --cli-input-json. The subshell keeps the variable out of the caller's shell.
locals {
  token_request = jsonencode({
    Name              = aws_cloudwatch_event_connection.github.name
    AuthorizationType = "API_KEY"
    AuthParameters = {
      ApiKeyAuthParameters = {
        ApiKeyName  = "Authorization"
        ApiKeyValue = ""
      }
      InvocationHttpParameters = {
        HeaderParameters = [
          for header in local.github_headers : {
            Key           = header.key
            Value         = header.value
            IsValueSecret = false
          }
        ]
      }
    }
  })

  set_token_command = replace(trimspace(<<-EOT
    ( umask 077; f="$(mktemp)" || exit 1; trap 'rm -f "$f"' EXIT;
    printf '%s' 'Fine-grained GitHub token for ${aws_cloudwatch_event_connection.github.name} (input hidden): ' >&2;
    IFS= read -rs GH_DISPATCH_TOKEN; echo >&2;
    [ -n "$GH_DISPATCH_TOKEN" ] || { echo 'No token given, nothing changed.' >&2; exit 1; };
    export GH_DISPATCH_TOKEN;
    jq -n --argjson req '${local.token_request}' '$req | .AuthParameters.ApiKeyAuthParameters.ApiKeyValue = "Bearer " + env.GH_DISPATCH_TOKEN' > "$f"
    && aws events update-connection --region ${local.region} --cli-input-json "file://$f" --query ConnectionState --output text > /dev/null
    && for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    s="$(aws events describe-connection --region ${local.region} --name ${aws_cloudwatch_event_connection.github.name} --query ConnectionState --output text)";
    case "$s" in UPDATING|AUTHORIZING) sleep 5 ;; *) break ;; esac; done
    && echo "$s" )
  EOT
  ), "\n", " ")

  test_dispatch_command = join(" ", [
    "aws events put-events --region ${local.region}",
    "--entries '${jsonencode([{ Source = local.test_event_source, DetailType = "manual dispatch test", Detail = "{}" }])}'",
    "--query FailedEntryCount --output text",
  ])
}

output "connection_name" {
  value = aws_cloudwatch_event_connection.github.name
}

output "connection_arn" {
  value = aws_cloudwatch_event_connection.github.arn
}

output "api_destination_arn" {
  value = aws_cloudwatch_event_api_destination.dispatch.arn
}

output "rule_names" {
  description = "Every dispatching rule, keyed by purpose."
  value       = { for key, rule in aws_cloudwatch_event_rule.dispatch : key => rule.name }
}

output "alarm_topic_arn" {
  value = aws_sns_topic.alarms.arn
}

output "dead_letter_queue_url" {
  description = "Read the ERROR_CODE and RULE_ARN attributes here after an alarm."
  value       = aws_sqs_queue.dlq.url
}

output "set_token_command" {
  description = "Prompts for the GitHub token and stores it in the connection. Contains no secret. Run with bash; prints the settled connection state, AUTHORIZED when it worked."
  value       = local.set_token_command
}

output "test_dispatch_command" {
  description = "Fires the test rule, which dispatches watchdog.yml through the same path as the schedules. Prints 0 when the event was accepted."
  value       = local.test_dispatch_command
}
