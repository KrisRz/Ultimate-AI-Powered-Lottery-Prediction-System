variable "region" {
  description = "Region for the bucket. UK audience, so London."
  type        = string
  default     = "eu-west-2"
}

variable "root_domain" {
  description = "Existing Route 53 hosted zone. Looked up, never created."
  type        = string
  default     = "krisgrzepka.com"
}

variable "site_domain" {
  description = "Where the explainer is published."
  type        = string
  default     = "lotto.krisgrzepka.com"
}

variable "bucket_name" {
  description = <<-EOT
    Origin bucket. Deliberately not the domain name, which is the convention
    used elsewhere in this account: CloudFront reaches a private
    Origin-Access-Control bucket over HTTPS at
    <bucket>.s3.<region>.amazonaws.com, and the wildcard certificate there
    matches a single label only. A bucket called lotto.krisgrzepka.com would
    fail TLS validation on every origin fetch.
  EOT
  type        = string
  default     = "lotto-ev-site-590183672693"
}

variable "github_repo" {
  description = "owner/name, used to pin the OIDC trust policy."
  type        = string
  default     = "KrisRz/Ultimate-AI-Powered-Lottery-Prediction-System"
}

variable "github_branch" {
  description = "The only branch allowed to deploy."
  type        = string
  default     = "main"
}

variable "create_oidc_provider" {
  description = <<-EOT
    The GitHub OIDC provider is account-global. This account has none today,
    so Terraform creates it. Flip to false if another stack ever claims it
    first, or apply will fail with EntityAlreadyExists.
  EOT
  type        = bool
  default     = true
}

variable "monthly_budget_usd" {
  description = <<-EOT
    Alarm, not a cap, and in USD because that is the unit AWS Budgets bills in.
    At portfolio traffic this stack costs pennies, so anything approaching this
    figure means something is wrong - a runaway invalidation loop, or an object
    being served uncached.
  EOT
  type        = number
  default     = 5
}

variable "budget_alert_email" {
  description = "Where the budget alarm goes."
  type        = string
  default     = "krisgrzepka@gmail.com"
}
