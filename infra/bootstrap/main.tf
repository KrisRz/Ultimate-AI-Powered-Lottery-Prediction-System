/**
 * State bucket. Run once, with local state, before `terraform init` in ../live.
 *
 *   cd infra/bootstrap && terraform init && terraform apply
 *
 * Its own state file stays on disk and is gitignored: the bucket that holds
 * every other state file cannot itself be tracked in one. Losing this state is
 * survivable - the bucket persists, and it can be imported back.
 */

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "lotto-ev-site"
      ManagedBy = "terraform"
    }
  }
}

variable "region" {
  type    = string
  default = "eu-west-2"
}

variable "bucket_name" {
  type    = string
  default = "lotto-ev-tfstate-590183672693"
}

resource "aws_s3_bucket" "state" {
  bucket = var.bucket_name

  lifecycle {
    # State is the one thing here that is not reproducible from the repo.
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    # Non-negotiable: versioning is how a corrupted or truncated state file is
    # recovered, and it is what native S3 locking assumes.
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "bucket_name" {
  value = aws_s3_bucket.state.id
}
