
resource "aws_sagemaker_feature_group" "customer_features" {
  feature_group_name             = "${var.project}-${var.environment}-customer-features"
  record_identifier_feature_name = "customer_id"
  event_time_feature_name        = "event_time"
  role_arn                       = var.data_engineer_role_arn

  description = "Northstar customer RFM and churn features"

  feature_definition {
    feature_name = "customer_id"
    feature_type = "String"
  }

  feature_definition {
    feature_name = "event_time"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "days_since_last_purchase"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "customer_tenure_days"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "purchase_frequency_30d"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "purchase_frequency_90d"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "purchase_frequency_180d"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "avg_order_value"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "total_spend_90d"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "total_lifetime_value"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "avg_basket_size_6m"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "category_diversity_score"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "online_to_store_ratio"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "loyalty_tier"
    feature_type = "String"
  }

  feature_definition {
    feature_name = "churn_risk_score"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "churn_label"
    feature_type = "Integral"
  }

  online_store_config {
    enable_online_store = true
  }

  offline_store_config {
    disable_glue_table_creation = false

    s3_storage_config {
      s3_uri = "s3://${var.bucket_name}/features/offline-store/"
    }

    data_catalog_config {
      catalog     = "AwsDataCatalog"
      database    = "${var.project}_${var.environment}_features"
      table_name  = "customer_features"
    }
  }

  tags = {
    Name = "${var.project}-${var.environment}-customer-features"
  }
}