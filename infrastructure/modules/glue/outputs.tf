output "feature_engineer_job_name" {
  description = "Name of the customer feature engineering Glue job"
  value       = aws_glue_job.feature_engineer.name
}

output "feature_engineer_job_arn" {
  description = "ARN of the customer feature engineering Glue job"
  value       = aws_glue_job.feature_engineer.arn
}