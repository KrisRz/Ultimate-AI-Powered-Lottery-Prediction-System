terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # State locking is native to the S3 backend from Terraform 1.10 - the
  # DynamoDB table this used to need is gone. Create the bucket with
  # infra/bootstrap before the first init here.
  backend "s3" {
    bucket       = "lotto-ev-tfstate-590183672693"
    key          = "site/terraform.tfstate"
    region       = "eu-west-2"
    encrypt      = true
    use_lockfile = true
  }
}
