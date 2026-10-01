resource "aws_glue_job" "feature_engineer" {
  name     = "${var.project}-${var.environment}-feature-engineer"
  role_arn = var.data_engineer_role_arn

  glue_version = "4.0"

  worker_type       = "G.1X"
  number_of_workers = 2

  timeout = 30

  command {
    name            = "glueetl"
    script_location = "s3://${var.bucket_name}/artifacts/glue/feature_engineer.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-glue-datacatalog"          = "true"
    "--enable-continuous-cloudwatch_log" = "true"
    "--input_path"                       = "s3://${var.bucket_name}/processed/customers/"
    "--output_path"                      = "s3://${var.bucket_name}/features/customers/"
    "--feature_group_name"               = var.feature_group_name
    "--region"                            = var.aws_region
  }

  execution_property {
    max_concurrent_runs = 1
  }

  tags = {
    Name = "${var.project}-${var.environment}-feature-engineer"
  }
}

resource "aws_s3_object" "feature_engineer_script" {
  bucket = var.bucket_name
  key    = "artifacts/glue/feature_engineer.py"
  source = var.feature_engineer_script_path

  etag = filemd5(var.feature_engineer_script_path)
}