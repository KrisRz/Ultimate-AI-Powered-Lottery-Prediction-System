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
