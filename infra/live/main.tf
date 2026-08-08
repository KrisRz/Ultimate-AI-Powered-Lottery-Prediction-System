# The hosted zone already exists and holds the portfolio's own records. Look it
# up; creating one would mint new nameservers and quietly break the domain.
data "aws_route53_zone" "root" {
  name         = "${var.root_domain}."
  private_zone = false
}

# --- certificate -------------------------------------------------------------
# In us-east-1 regardless of where anything else lives: CloudFront reads
# certificates from that region only. Kept here rather than in a module because
# validation needs records in the zone above, so splitting it would just pass
# the same values back and forth.

resource "aws_acm_certificate" "site" {
  provider = aws.us_east_1

  domain_name       = var.site_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for option in aws_acm_certificate.site.domain_validation_options :
    option.domain_name => {
      name   = option.resource_record_name
      record = option.resource_record_value
      type   = option.resource_record_type
    }
  }

  zone_id         = data.aws_route53_zone.root.zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}

# Blocks until DNS propagates. The first apply can sit here for several
# minutes; that is the certificate authority answering, not Terraform hanging.
resource "aws_acm_certificate_validation" "site" {
  provider = aws.us_east_1

  certificate_arn         = aws_acm_certificate.site.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

# --- the site ----------------------------------------------------------------

module "site" {
  source = "../modules/static-site"

  bucket_name     = var.bucket_name
  site_domain     = var.site_domain
  certificate_arn = aws_acm_certificate_validation.site.certificate_arn
}

# Both records. An AAAA-less site is invisible to IPv6-only clients, and the
# distribution has IPv6 enabled anyway.
resource "aws_route53_record" "site" {
  for_each = toset(["A", "AAAA"])

  zone_id = data.aws_route53_zone.root.zone_id
  name    = var.site_domain
  type    = each.value

  alias {
    name    = module.site.distribution_domain_name
    zone_id = module.site.distribution_hosted_zone_id
    # The alias target's own health, not a health check we would have to run.
    evaluate_target_health = false
  }
}

# --- deploy identity ---------------------------------------------------------

module "github_oidc" {
  source = "../modules/github-oidc"

  create_provider  = var.create_oidc_provider
  github_repo      = var.github_repo
  github_branch    = var.github_branch
  name_prefix      = "lotto-ev-site"
  bucket_arn       = module.site.bucket_arn
  distribution_arn = module.site.distribution_arn
}

# --- cost alarm --------------------------------------------------------------
# This stack should cost pennies: CloudFront's free allowance covers portfolio
# traffic, the origin is a few megabytes, and there is no compute at all. An
# alert here means something is wrong rather than popular - most likely an
# object being served uncached, or an invalidation loop.

resource "aws_budgets_budget" "site" {
  provider = aws.us_east_1

  name         = "lotto-ev-site-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$lotto-ev-site"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
