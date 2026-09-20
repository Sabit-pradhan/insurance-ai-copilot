/* ============================================================
   INSURANCE AI COPILOT PROJECT
   FILE: 01_constraints_indexes.sql
   PURPOSE:
   1. Primary Keys
   2. Foreign Keys
   3. Indexes
   ============================================================ */


/* ============================================================
   IMPORTANT:
   If Primary Keys were already created while creating tables,
   DO NOT run the PRIMARY KEY section again.
   ============================================================ */


/* ============================================================
   1. PRIMARY KEYS
   ============================================================ */

-- Run these ONLY if PKs are not already present.

ALTER TABLE customers
ADD CONSTRAINT pk_customers
PRIMARY KEY (customer_id);


ALTER TABLE agents
ADD CONSTRAINT pk_agents
PRIMARY KEY (agent_id);


ALTER TABLE policies
ADD CONSTRAINT pk_policies
PRIMARY KEY (policy_id);


ALTER TABLE claims
ADD CONSTRAINT pk_claims
PRIMARY KEY (claim_id);


ALTER TABLE premium_transactions
ADD CONSTRAINT pk_premium_transactions
PRIMARY KEY (transaction_id);



/* ============================================================
   2. FOREIGN KEYS
   ============================================================ */

-- Customer -> Policies
ALTER TABLE policies
ADD CONSTRAINT fk_policies_customer
FOREIGN KEY (customer_id)
REFERENCES customers(customer_id);


-- Agent -> Policies
ALTER TABLE policies
ADD CONSTRAINT fk_policies_agent
FOREIGN KEY (agent_id)
REFERENCES agents(agent_id);


-- Policy -> Claims
ALTER TABLE claims
ADD CONSTRAINT fk_claims_policy
FOREIGN KEY (policy_id)
REFERENCES policies(policy_id);


-- Policy -> Premium Transactions
ALTER TABLE premium_transactions
ADD CONSTRAINT fk_premium_policy
FOREIGN KEY (policy_id)
REFERENCES policies(policy_id);


-- Policy -> Renewals
ALTER TABLE renewals
ADD CONSTRAINT fk_renewals_policy
FOREIGN KEY (policy_id)
REFERENCES policies(policy_id);


-- Policy -> Underwriting
ALTER TABLE underwriting
ADD CONSTRAINT fk_underwriting_policy
FOREIGN KEY (policy_id)
REFERENCES policies(policy_id);



/* ============================================================
   3. INDEXES
   ============================================================ */

-- Faster customer-policy lookup
CREATE INDEX IF NOT EXISTS idx_policies_customer
ON policies(customer_id);


-- Faster agent-policy lookup
CREATE INDEX IF NOT EXISTS idx_policies_agent
ON policies(agent_id);


-- Faster claim lookup using policy
CREATE INDEX IF NOT EXISTS idx_claims_policy
ON claims(policy_id);


-- Faster premium transaction lookup using policy
CREATE INDEX IF NOT EXISTS idx_premium_policy
ON premium_transactions(policy_id);


-- Faster renewal lookup using policy
CREATE INDEX IF NOT EXISTS idx_renewals_policy
ON renewals(policy_id);


-- Faster underwriting lookup using policy
CREATE INDEX IF NOT EXISTS idx_underwriting_policy
ON underwriting(policy_id);