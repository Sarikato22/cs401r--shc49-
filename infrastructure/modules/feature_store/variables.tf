variable "project" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "bucket_name" {
  description = "S3 data bucket containing the Feature Store offline path"
  type        = string
}

variable "data_engineer_role_arn" {
  description = "IAM role ARN used by SageMaker Feature Store"
  type        = string
}