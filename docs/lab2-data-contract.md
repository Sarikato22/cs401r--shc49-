## Data Contract: processed/customers

### Producer

Team / process: Glue ETL job `northstar-dev-transform`

### Consumers

* Feature engineering job `northstar-dev-feature-engineer`
* (Future) Direct model training in Lab 3

### Grain

One row per transaction. A customer appears on many rows.

### Schema

| Column            | Type                   | Nullable | Description                                                                             |
| ----------------- | ---------------------- | -------- | --------------------------------------------------------------------------------------- |
| `transaction_id`  | String                 | No       | Unique identifier for the transaction.                                                  |
| `customer_id`     | String                 | No       | Unique identifier for the customer. A customer may appear on multiple transaction rows. |
| `purchase_date`   | String (ISO 8601 date) | No       | Date on which the transaction occurred, formatted as `YYYY-MM-DD`.                      |
| `purchase_amount` | Double                 | No       | Monetary value of the transaction. Must be non-negative.                                |
| `quantity`        | Integer                | No       | Number of items included in the transaction. Must be a positive integer.                |
| `channel`         | String                 | No       | Purchase channel, such as online or store.                                              |
| `category`        | String                 | No       | Product category associated with the transaction.                                       |

### Quality Guarantees

* `customer_id` is never null or empty.
* `transaction_id` is never null or empty, and no duplicate `transaction_id` rows are present.
* `purchase_amount` is numeric and greater than or equal to `0`.
* `quantity` is an integer greater than `0`.
* `purchase_date` is a valid ISO 8601 date in `YYYY-MM-DD` format.
* `channel` is populated with a recognized purchase channel.
* `category` is populated; unknown or unavailable categories are represented consistently rather than as null values.
* A `customer_id` appearing on multiple rows is expected because the dataset has transaction-level grain.

### SLA

Data is available in `processed/customers/` within 2 hours of landing in `raw/customers/`.

### Versioning

* Schema changes require a new S3 prefix (e.g., `processed/customers/v2/`).
* Breaking changes require consumer notification 5 business days in advance.
* Existing consumers must continue to receive the contracted schema for the duration of the applicable version.
