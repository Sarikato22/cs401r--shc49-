# ── environments/dev ─────────────────────────────────────────────────────────
# Wire the four modules together here. Each module call passes var.project and
# var.environment down; nothing in modules/ hardcodes a name.
#
# Uncomment each block as you implement the module it calls.

module "vpc" {
  source              = "../../modules/vpc"
  project             = var.project
  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidr  = var.public_subnet_cidr
  availability_zone   = var.availability_zone
  private_subnet_cidr = var.private_subnet_cidr
  enable_nat_gateway  = true
}

module "storage" {
  source                 = "../../modules/storage"
  project                = var.project
  environment            = var.environment
  enable_lifecycle_rules = true
}

module "iam" {
  source      = "../../modules/iam"
  project     = var.project
  environment = var.environment
}

module "sagemaker" {
  source             = "../../modules/sagemaker"
  project            = var.project
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = [module.vpc.private_subnet_id]
  security_group_ids = [module.vpc.security_group_id]
  execution_role_arn = module.iam.ml_engineer_role_arn
  instance_type      = var.sagemaker_instance_type
}

module "feature_store" {
  source                 = "../../modules/feature_store"
  project                = var.project
  environment            = var.environment
  bucket_name            = module.storage.bucket_name
  data_engineer_role_arn = module.iam.data_engineer_role_arn
}
module "glue" {
  source                       = "../../modules/glue"
  project                      = var.project
  environment                  = var.environment
  bucket_name                  = module.storage.bucket_name
  data_engineer_role_arn       = module.iam.data_engineer_role_arn
  feature_group_name           = module.feature_store.feature_group_name
  feature_engineer_script_path = "${path.root}/../../../glue-scripts/feature_engineer.py"
  transform_script_path        = "${path.root}/../../../glue-scripts/transform.py"
  aws_region                   = var.aws_region
}