# These three go into the repo's Actions variables, not its secrets. A bucket
# name, a distribution id and a role ARN are not credentials - the role is
# useless without a workflow token from this exact repo and branch - and using
# `vars` means a fork fails loudly instead of silently deploying somewhere.
#
#   gh variable set SITE_BUCKET          --body "$(terraform output -raw bucket_name)"
#   gh variable set SITE_DISTRIBUTION_ID --body "$(terraform output -raw distribution_id)"
#   gh variable set SITE_DOMAIN          --body "$(terraform output -raw site_domain)"
#   gh variable set AWS_DEPLOY_ROLE_ARN  --body "$(terraform output -raw deploy_role_arn)"

output "bucket_name" {
  value = module.site.bucket_name
}

output "distribution_id" {
  value = module.site.distribution_id
}

output "site_domain" {
  value = var.site_domain
}

output "deploy_role_arn" {
  value = module.github_oidc.deploy_role_arn
}

output "distribution_domain_name" {
  description = "Reachable immediately, before the alias records propagate."
  value       = module.site.distribution_domain_name
}

# --- draw trigger ------------------------------------------------------------
# None of these is a secret. The token goes in through set_token_command,
# which prompts for it and never writes it anywhere Terraform can see:
#
#   bash -c "$(terraform output -raw set_token_command)"

output "draw_trigger_connection_name" {
  value = module.draw_trigger.connection_name
}

output "draw_trigger_connection_arn" {
  value = module.draw_trigger.connection_arn
}

output "draw_trigger_rule_names" {
  value = module.draw_trigger.rule_names
}

output "draw_trigger_alarm_topic_arn" {
  value = module.draw_trigger.alarm_topic_arn
}

output "draw_trigger_dead_letter_queue_url" {
  value = module.draw_trigger.dead_letter_queue_url
}

output "set_token_command" {
  value = module.draw_trigger.set_token_command
}

output "test_dispatch_command" {
  value = module.draw_trigger.test_dispatch_command
}
