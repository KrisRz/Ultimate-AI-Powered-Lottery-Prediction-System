provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "lotto-ev-site"
      ManagedBy = "terraform"
      Repo      = var.github_repo
    }
  }
}

# CloudFront only accepts certificates from us-east-1, whatever region the
# bucket lives in. Budgets is likewise a global service that answers there.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project   = "lotto-ev-site"
      ManagedBy = "terraform"
      Repo      = var.github_repo
    }
  }
}
