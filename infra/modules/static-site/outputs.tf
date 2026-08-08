output "bucket_name" {
  description = "Origin bucket, for the deploy workflow's s3 sync."
  value       = aws_s3_bucket.site.id
}

output "bucket_arn" {
  value = aws_s3_bucket.site.arn
}

output "distribution_id" {
  description = "For create-invalidation."
  value       = aws_cloudfront_distribution.site.id
}

output "distribution_arn" {
  value = aws_cloudfront_distribution.site.arn
}

output "distribution_domain_name" {
  description = "The dxxxx.cloudfront.net name, reachable before DNS propagates."
  value       = aws_cloudfront_distribution.site.domain_name
}

output "distribution_hosted_zone_id" {
  description = "For the Route 53 alias. Never hardcode Z2FDTNDATAQYW2."
  value       = aws_cloudfront_distribution.site.hosted_zone_id
}
