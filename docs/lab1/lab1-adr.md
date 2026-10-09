## ADR-001: NorthStar Platform Foundation

### Status

Accepted

### Context

NorthStar is building an AI infrastructure that will eventually support several connected solutions, including churn prediction, personalized offer generation, and a customer service agent. I think it is important to view these as parts of one platform rather than three completely separate projects. For example, a customer might contact customer service several times because they are having the same problem. That information could be useful when determining whether that customer is likely to churn. If the customer is considered high risk, NorthStar could then use the offer generation system to create a more appropriate retention offer.

Because of this, NorthStar needs infrastructure that can support different AI systems while giving them a common foundation. The goal of this lab was not to build the actual churn model or customer service agent yet. Instead, it was to establish some of the AWS infrastructure that those future systems can use.

### Decision

The infrastructure for this lab was built around the foundation NorthStar will need as its AI platform grows. 

The VPC and public subnet provide the basic network environment for the platform. The VPC uses a `10.0.0.0/16` address range and the public subnet uses `10.0.100.0/24`. The internet gateway and route table allow the subnet to have access to the internet. For this lab, one public subnet was enough because we are working with a small development environment. 

The S3 bucket provides shared storage for the platform. It has separate areas for raw data, processed data, features, and model artifacts. we separated these because the platform will have different types of data at different stages, and keeping them organized makes it easier for the three systems to use the same storage. This makes sense for NorthStar because the churn system will eventually need to turn customer information into features and produce model results. Storing the data in its most raw state will allow for multiple uses.

The S3 bucket also has public access blocked, versioning enabled, and encryption enabled, given the sensitivity of the data(customer data).

The IAM role provides the identity that SageMaker can use when performing its work. The role is intentionally limited instead of giving it broad access to AWS. This is important because SageMaker can be an expensive service. Giving an ML engineer unlimited permissions could make it easier to accidentally create or use resources that cost money. The role therefore gives access to the actions needed for the lab without allowing it to write to the raw data area.

The SageMaker Domain and User Profile then provide the actual ML development environment. Instead of having the ML environment completely separate from the rest of the infrastructure, SageMaker uses the tools we previously described.

### Consequences

#### What this makes easy

One shared platform makes it easier for NorthStar's AI systems to work with related customer information. The churn model can eventually use customer behavior and other information to produce weekly 90-day churn scores, while the offer generation and customer service systems can use related information to respond to those customers.

The four S3 storage areas also give the platform a predictable organization for data as it moves through different stages. Instead of every system creating its own unrelated storage structure, they have a common starting point.

Using Terraform also makes it much easier to reproduce the infrastructure. During this lab I found the AWS console easier to understand personally, but Terraform became more valuable when I think about doing this at a larger scale. Instead of manually clicking through AWS and hoping everything is configured the same way, Terraform can create the infrastructure from the same configuration and allow me to check the quality of the work just by inspecting a file.

#### What this makes harder

Terraform has a steeper learning curve than using the AWS console. With the console, I can see resources and click through the setup, while Terraform requires understanding configuration files, variables, modules, dependencies, and state.

The single public subnet is also a limitation. It is enough for this small development environment, but a larger NorthStar platform could need additional subnets and stronger network separation.

Using shared infrastructure also means that changes can affect multiple systems. If NorthStar eventually gives the churn, offer generation, and customer service systems very different security requirements, the shared IAM and storage design may need to become more separated. 

#### What would cause you to revisit this decision

I would revisit this architecture if NorthStar moved from a small development environment toward production-scale workloads. For example, needing multiple availability zones, private networking, much larger amounts of data, or separate permissions for the three AI systems would require additional design.

I would also revisit the IAM model if one system needed permissions that the other systems should not have. Since NorthStar is dealing with customer information and has regulatory requirements around that data, permissions would need to become more specific as the platform grows.

### Alternative Considered

One alternative would be to build separate infrastructure for each of NorthStar's three AI systems. For example, the churn prediction system could have its own storage and permissions, while the offer generation and customer service systems could each have their own infrastructure as well. This would provide more separation between the systems and could make sense if they eventually have very different security or data requirements.

However, I think this only makes sense if the data is unrelated. For example, if we were asked to do the churn and then something about internal processes, there would be no reason for those two to share information.

### AWS Service Selection

* **Networking isolation model:** Amazon VPC because NorthStar needs a defined network environment for its shared AI platform, starting with one public subnet for this development lab.
* **Storage design:** Amazon S3 because the platform needs shared storage that can organize raw data, processed data, features, and model artifacts.
* **Identity model:** AWS IAM because SageMaker needs a controlled identity with only the permissions required for its ML work rather than unrestricted AWS access.
* **ML development environment:** Amazon SageMaker because NorthStar needs a managed environment for developing and eventually training its machine-learning systems.
