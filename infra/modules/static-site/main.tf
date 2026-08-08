terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

data "aws_caller_identity" "current" {}

# --- origin ------------------------------------------------------------------

resource "aws_s3_bucket" "site" {
  bucket = var.bucket_name
}

# The bucket is never reachable directly. Every byte is served through
# CloudFront, which is what makes the response headers and the 403->404
# mapping below unavoidable rather than advisory.
resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "site" {
  bucket = aws_s3_bucket.site.id

  rule {
    # Disables ACLs entirely. The deploy role therefore needs no
    # s3:PutObjectAcl, and sending one would be an error rather than a no-op.
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id

  rule {
    apply_server_side_encryption_by_default {
      # SSE-S3, not KMS. The content is a public web page, so a customer key
      # protects nothing, while costing a key policy for the distribution and
      # a per-request charge on every origin fetch.
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "site" {
  bucket = aws_s3_bucket.site.id

  versioning_configuration {
    # A bad deploy is `git revert` plus a rebuild, but versioning makes the
    # previous objects recoverable in the minutes before that lands.
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "site" {
  bucket     = aws_s3_bucket.site.id
  depends_on = [aws_s3_bucket_versioning.site]

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Note: no aws_s3_bucket_website_configuration. The website endpoint is
# HTTP-only and cannot be private, which is the whole reason for the
# viewer-request function that resolves index documents instead.

# --- distribution ------------------------------------------------------------

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = var.bucket_name
  description                       = "Signed access from CloudFront to the private origin"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_function" "rewrite" {
  name    = "${replace(var.site_domain, ".", "-")}-rewrite"
  runtime = "cloudfront-js-2.0"
  comment = "Resolve directory URLs to index.html on the S3 REST endpoint"
  publish = true
  code    = file("${path.module}/functions/rewrite.js")
}

resource "aws_cloudfront_response_headers_policy" "site" {
  name    = "${replace(var.site_domain, ".", "-")}-headers"
  comment = "Security headers for the static explainer"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = true
      preload                    = false
      override                   = true
    }

    content_type_options {
      override = true
    }

    frame_options {
      frame_option = "DENY"
      override     = true
    }

    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }

    content_security_policy {
      override = true
      # script-src needs 'unsafe-inline': Next's static export inlines a
      # bootstrap script, and a nonce would need a server to generate it.
      # Everything else is locked to self - no third-party host is contacted,
      # which is also why the fonts are self-hosted at build time.
      content_security_policy = join("; ", [
        "default-src 'none'",
        "img-src 'self' data:",
        "style-src 'self' 'unsafe-inline'",
        "script-src 'self' 'unsafe-inline'",
        "font-src 'self'",
        "connect-src 'self'",
        "manifest-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
      ])
    }
  }
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = var.site_domain
  aliases             = [var.site_domain]
  price_class         = var.price_class
  default_root_object = "index.html"

  origin {
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_id                = "s3-${var.bucket_name}"
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  default_cache_behavior {
    target_origin_id       = "s3-${var.bucket_name}"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # Managed-CachingOptimized. Respects the Cache-Control the deploy sets on
    # each object, which is how hashed assets get a year and HTML gets none.
    cache_policy_id            = "658327ea-f89d-4fab-a63d-7e88639e58f6"
    response_headers_policy_id = aws_cloudfront_response_headers_policy.site.id

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.rewrite.arn
    }
  }

  # S3 answers a missing key with 403, not 404, because the policy below grants
  # GetObject but deliberately not ListBucket. Both have to be mapped, or every
  # typo would surface as a bare Access Denied.
  #
  # Mapping these to 200 /index.html is the single-page-app pattern and would
  # be wrong here: a static export has real URLs, and answering 200 for a
  # missing one tells search engines the typo is a page.
  custom_error_response {
    error_code            = 403
    response_code         = 404
    response_page_path    = "/404/index.html"
    error_caching_min_ttl = 60
  }

  custom_error_response {
    error_code            = 404
    response_code         = 404
    response_page_path    = "/404/index.html"
    error_caching_min_ttl = 60
  }

  viewer_certificate {
    acm_certificate_arn      = var.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
}

# Separate resource, never an inline policy on the bucket: the policy has to
# name the distribution ARN, and the distribution has to name the bucket, so
# inlining creates a genuine dependency cycle.
data "aws_iam_policy_document" "bucket" {
  statement {
    sid       = "AllowCloudFrontRead"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.site.arn}/*"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.site.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket     = aws_s3_bucket.site.id
  policy     = data.aws_iam_policy_document.bucket.json
  depends_on = [aws_s3_bucket_public_access_block.site]
}
