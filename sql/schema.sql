CREATE DATABASE IF NOT EXISTS insurance_fraud_db;
USE insurance_fraud_db;

DROP TABLE IF EXISTS insurance_applications;

CREATE TABLE insurance_applications (
    application_id VARCHAR(30) PRIMARY KEY,
    customer_age INT,
    region VARCHAR(100),
    vehicle_type VARCHAR(100),
    vehicle_age INT,
    contract_type VARCHAR(100),
    application_channel VARCHAR(100),
    occupation VARCHAR(100),
    premium_amount DECIMAL(12,2),
    claim_amount DECIMAL(12,2),
    num_claims_12m INT,
    num_claims_total INT,
    payment_delay_count INT,
    previous_cancellations INT,
    address_changes_12m INT,
    policy_changes_3m INT,
    contract_age_months INT,
    accident_report_delay_days INT,
    application_hour INT,
    fraud_label INT,
    claims_to_premium_ratio DECIMAL(12,3),
    short_contract_large_claim INT,
    high_claim_frequency_flag INT,
    high_payment_delay_flag INT,
    multiple_recent_changes_flag INT,
    created_at DATE
);