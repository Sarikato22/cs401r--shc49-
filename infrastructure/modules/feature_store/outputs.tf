output "feature_group_name" {
  description = "Customer Feature Store feature group name"
  value       = aws_sagemaker_feature_group.customer_features.feature_group_name
}

output "feature_group_arn" {
  description = "Customer Feature Store feature group ARN"
  value       = aws_sagemaker_feature_group.customer_features.arn
}