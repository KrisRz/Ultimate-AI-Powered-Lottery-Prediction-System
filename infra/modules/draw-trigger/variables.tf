variable "name_prefix" {
  description = "Prefix for every name in the module."
  type        = string
}

variable "github_repo" {
  description = "owner/name whose workflows are dispatched."
  type        = string
}

variable "git_ref" {
  description = "The ref each dispatched run checks out."
  type        = string
  default     = "main"
}

variable "alarm_email" {
  description = "Receives an email when a dispatch fails. AWS mails a confirmation link first; nothing arrives until it is clicked."
  type        = string
}

# The three schedules are UTC, like GitHub's own cron, so the two sets stay
# comparable line by line. The UK draw is at about 20:00 local, which is
# 19:00 UTC in summer and 20:00 in winter; 21:50 clears both.

variable "collect_evening_schedule" {
  description = <<-EOT
    Right after the Wed/Sat draw. Five minutes behind GitHub's own 21:45 cron,
    which stays in collect.yml as a fallback: ingestion is idempotent, so
    whichever of the two runs comes second adds nothing.
  EOT
  type        = string
  default     = "cron(50 21 ? * WED,SAT *)"

  validation {
    condition     = can(regex("^(cron|rate)\\(.+\\)$", var.collect_evening_schedule))
    error_message = "An EventBridge schedule expression: cron(...) or rate(...)."
  }
}

variable "collect_retry_schedule" {
  description = "Next-morning retry, for a draw whose data was not published by the evening run."
  type        = string
  default     = "cron(5 6 ? * THU,SUN *)"

  validation {
    condition     = can(regex("^(cron|rate)\\(.+\\)$", var.collect_retry_schedule))
    error_message = "An EventBridge schedule expression: cron(...) or rate(...)."
  }
}

variable "watchdog_schedule" {
  description = "The check that the draw actually landed, after the retry has had its chance."
  type        = string
  default     = "cron(5 12 ? * THU,SUN *)"

  validation {
    condition     = can(regex("^(cron|rate)\\(.+\\)$", var.watchdog_schedule))
    error_message = "An EventBridge schedule expression: cron(...) or rate(...)."
  }
}
