terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# GitHub's OIDC endpoint, federated so Actions can exchange a short-lived
# workflow token for AWS credentials. No access key exists to leak, rotate, or
# find in a repo six months later.
#
# The provider is one per account. This account has none today, hence the
# default of creating it; the data-source branch is what stops a second stack
# from failing with EntityAlreadyExists.
resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  provider_arn = var.create_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

data "aws_iam_policy_document" "trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # StringEquals on the exact ref, never StringLike with a wildcard. With
    # `repo:owner/name:*` any branch - including one opened by a fork's pull
    # request workflow - could assume this role and publish to production.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.name_prefix}-deploy"
  description          = "Publishes the built site. Cannot change the infrastructure that serves it."
  assume_role_policy   = data.aws_iam_policy_document.trust.json
  max_session_duration = 3600
}

# Exactly what `aws s3 sync --delete` and `create-invalidation` need, and
# nothing else. In particular this role cannot touch the bucket policy, the
# distribution config, or any other bucket in the account - so a compromised
# workflow can deface the page but not repoint it or read anything else.
data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "ListTheOriginBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.bucket_arn]
  }

  statement {
    sid    = "WriteObjects"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:DeleteObject",
    ]
    resources = ["${var.bucket_arn}/*"]
  }

  statement {
    sid    = "InvalidateThisDistribution"
    effect = "Allow"
    actions = [
      "cloudfront:CreateInvalidation",
      "cloudfront:GetInvalidation",
    ]
    # Resource-level scoping, supported for invalidations since 2021. The
    # lazy alternative is "*", which would let this role bill the account by
    # invalidating every distribution it can name.
    resources = [var.distribution_arn]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "publish"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

# No second, broader role for Terraform itself. Infrastructure changes are
# applied from a laptop with a human present; handing CI the power to rewrite
# the bucket policy would undo the point of scoping the deploy role at all.
