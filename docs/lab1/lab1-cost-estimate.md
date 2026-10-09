# Monthly Cost Estimate

| Component             | Monthly Estimate | Key Assumptions                                     | One Optimization                                          |
| --------------------- | ---------------: | --------------------------------------------------- | --------------------------------------------------------- |
| SageMaker Studio      |        **$9.60** | 1 `ml.t3.medium` instance, 4 hrs/day, 20 days/month | Shut down Studio when not in use                          |
| S3 storage            |        **$2.30** | 100 GB of S3 Standard storage                       | Use lifecycle rules to move older data to cheaper storage |
| Internet Gateway      |        **$1.00** | Approx. 100 GB/month of internet traffic            | Reduce unnecessary data transfer                          |
| DynamoDB (state lock) |        **$0.05** | On-demand, low-volume Terraform state-lock requests | Keep the table limited to Terraform locking               |
| S3 state bucket       |        **$0.02** | 1 GB of S3 Standard storage                         | Keep unnecessary state versions from accumulating         |
| **Total**             |       **$12.97** | Small development environment                       | Shut down unused resources                                |
