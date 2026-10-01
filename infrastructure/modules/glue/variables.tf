variable "project" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "bucket_name" {
  description = "S3 data bucket used by the Glue job"
  type        = string
}

variable "data_engineer_role_arn" {
  description = "IAM role ARN assumed by the Glue job"
  type        = string
}

variable "feature_group_name" {
  description = "SageMaker Feature Group receiving the generated features"
  type        = string
}

variable "feature_engineer_script_path" {
  description = "Local path to the feature engineering Glue script"
  type        = string
}

variable "aws_region" {
  description = "AWS region where the Glue job runs"
  type        = string
}