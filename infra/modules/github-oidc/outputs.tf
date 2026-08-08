output "deploy_role_arn" {
  description = "Assumed by the deploy workflow. Publish as a repo variable, not a secret - it is not one."
  value       = aws_iam_role.deploy.arn
}

output "provider_arn" {
  value = local.provider_arn
}
