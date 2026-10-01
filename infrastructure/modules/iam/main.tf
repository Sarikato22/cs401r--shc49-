# ── modules/iam ──────────────────────────────────────────────────────────────
# Required resources (Task B1). Exactly one of each:
#
#   aws_iam_role                     MLEngineer, trusted by sagemaker.amazonaws.com
#   aws_iam_policy
#   aws_iam_role_policy_attachment
#
# Least privilege is graded in later labs, so start narrow: grant only the S3
# prefixes and SageMaker actions this role actually needs. A wildcard policy
# here will cost you points in Lab 2.

data "aws_caller_identity" "current" {}

locals {
  data_bucket_name = "${var.project}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
}

#MLEngineer 

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "ml_engineer" {
  name               = "${var.project}-${var.environment}-MLEngineer"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json

  tags = {
    Name = "${var.project}-${var.environment}-MLEngineer"
  }
}

resource "aws_iam_policy" "ml_engineer" {
  name = "${var.project}-${var.environment}-MLEngineer-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "sagemaker:CreateTrainingJob"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ml_engineer" {
  role       = aws_iam_role.ml_engineer.name
  policy_arn = aws_iam_policy.ml_engineer.arn
}

# Data Engineer

data "aws_iam_policy_document" "data_engineer_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type = "Service"
      identifiers = [
        "glue.amazonaws.com",
        "lambda.amazonaws.com",
        "sagemaker.amazonaws.com"
      ]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "data_engineer" {
  name               = "${var.project}-${var.environment}-DataEngineer"
  assume_role_policy = data.aws_iam_policy_document.data_engineer_assume_role.json

  tags = {
    Name = "${var.project}-${var.environment}-DataEngineer"
  }
}

resource "aws_iam_policy" "data_engineer" {
  name = "${var.project}-${var.environment}-DataEngineer-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "glue:*"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeVpcs",
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${local.data_bucket_name}/raw/*",
          "arn:aws:s3:::${local.data_bucket_name}/processed/*",
          "arn:aws:s3:::${local.data_bucket_name}/features/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObjectAcl"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}/features/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketAcl"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}"

        Condition = {
          StringLike = {
            "s3:prefix" = [
              "raw/*",
              "processed/*",
              "features/*"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}/artifacts/glue/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}"

        Condition = {
          StringLike = {
            "s3:prefix" = [
              "artifacts/glue/*"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:PutRecord",
          "sagemaker:CreateFeatureGroup",
          "sagemaker:DescribeFeatureGroup"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "data_engineer" {
  role       = aws_iam_role.data_engineer.name
  policy_arn = aws_iam_policy.data_engineer.arn
}

# Model Monitor

data "aws_iam_policy_document" "model_monitor_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "model_monitor" {
  name               = "${var.project}-${var.environment}-ModelMonitor"
  assume_role_policy = data.aws_iam_policy_document.model_monitor_assume_role.json

  tags = {
    Name = "${var.project}-${var.environment}-ModelMonitor"
  }
}

resource "aws_iam_policy" "model_monitor" {
  name = "${var.project}-${var.environment}-ModelMonitor-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:ListProcessingJobs",
          "sagemaker:DescribeProcessingJob"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}/artifacts/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = "arn:aws:s3:::${local.data_bucket_name}"

        Condition = {
          StringLike = {
            "s3:prefix" = [
              "artifacts/*"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "model_monitor" {
  role       = aws_iam_role.model_monitor.name
  policy_arn = aws_iam_policy.model_monitor.arn
}
