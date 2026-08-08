variable "create_provider" {
  description = "The provider is account-global; create it only if nothing else has."
  type        = bool
  default     = true
}

variable "github_repo" {
  description = "owner/name"
  type        = string
}

variable "github_branch" {
  description = "The single branch allowed to assume the deploy role."
  type        = string
  default     = "main"
}

variable "name_prefix" {
  description = "Role name prefix."
  type        = string
}

variable "bucket_arn" {
  type = string
}

variable "distribution_arn" {
  type = string
}
