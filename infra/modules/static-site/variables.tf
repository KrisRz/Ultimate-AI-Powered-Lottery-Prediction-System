variable "bucket_name" {
  description = "Origin bucket. Must contain no dots - see the note in live/variables.tf."
  type        = string

  validation {
    condition     = !strcontains(var.bucket_name, ".")
    error_message = "A dotted bucket name breaks TLS on the CloudFront-to-S3 origin fetch, because the *.s3.<region>.amazonaws.com certificate matches one label only."
  }
}

variable "site_domain" {
  description = "The alias CloudFront serves."
  type        = string
}

variable "certificate_arn" {
  description = "ACM certificate for site_domain. Must live in us-east-1."
  type        = string
}

variable "price_class" {
  description = "PriceClass_100 is North America and Europe, which is where the readers are."
  type        = string
  default     = "PriceClass_100"
}
