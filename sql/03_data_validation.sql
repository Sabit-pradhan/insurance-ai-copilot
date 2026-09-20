/* ============================================================
   INSURANCE PROJECT
   FILE: 03_data_validation.sql

   PURPOSE:
   1. Row count validation
   2. Duplicate ID validation
   3. NULL ID validation
   4. Relationship / orphan validation
   ============================================================ */


/* ============================================================
   1. CHECK TOTAL ROWS
   ============================================================ */

SELECT COUNT(*) AS total_customers
FROM customers;

SELECT COUNT(*) AS total_agents
FROM agents;

SELECT COUNT(*) AS total_policies
FROM policies;

SELECT COUNT(*) AS total_claims
FROM claims;

SELECT COUNT(*) AS total_premiums
FROM premiums;

SELECT COUNT(*) AS total_renewals
FROM renewals;

SELECT COUNT(*) AS total_underwriting
FROM underwriting;


/* ============================================================
   2. CHECK DUPLICATE PRIMARY IDs
   Expected Result: 0 rows
   ============================================================ */

-- Customers
SELECT
    customer_id,
    COUNT(*) AS duplicate_count
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;


-- Agents
SELECT
    agent_id,
    COUNT(*) AS duplicate_count
FROM agents
GROUP BY agent_id
HAVING COUNT(*) > 1;


-- Policies
SELECT
    policy_id,
    COUNT(*) AS duplicate_count
FROM policies
GROUP BY policy_id
HAVING COUNT(*) > 1;


-- Claims
SELECT
    claim_id,
    COUNT(*) AS duplicate_count
FROM claims
GROUP BY claim_id
HAVING COUNT(*) > 1;


-- Premiums
SELECT
    premium_txn_id,
    COUNT(*) AS duplicate_count
FROM premiums
GROUP BY premium_txn_id
HAVING COUNT(*) > 1;


-- Renewals
SELECT
    renewal_id,
    COUNT(*) AS duplicate_count
FROM renewals
GROUP BY renewal_id
HAVING COUNT(*) > 1;


-- Underwriting
SELECT
    underwriting_id,
    COUNT(*) AS duplicate_count
FROM underwriting
GROUP BY underwriting_id
HAVING COUNT(*) > 1;


/* ============================================================
   3. CHECK NULL PRIMARY IDs
   Expected Result: 0
   ============================================================ */

SELECT COUNT(*) AS null_customer_ids
FROM customers
WHERE customer_id IS NULL;


SELECT COUNT(*) AS null_agent_ids
FROM agents
WHERE agent_id IS NULL;


SELECT COUNT(*) AS null_policy_ids
FROM policies
WHERE policy_id IS NULL;


SELECT COUNT(*) AS null_claim_ids
FROM claims
WHERE claim_id IS NULL;


SELECT COUNT(*) AS null_premium_ids
FROM premiums
WHERE premium_txn_id IS NULL;


SELECT COUNT(*) AS null_renewal_ids
FROM renewals
WHERE renewal_id IS NULL;


SELECT COUNT(*) AS null_underwriting_ids
FROM underwriting
WHERE underwriting_id IS NULL;


/* ============================================================
   4. ORPHAN POLICY CHECK
   Policy customer must exist in customers table
   Expected Result: 0 rows
   ============================================================ */

SELECT
    p.policy_id,
    p.customer_id
FROM policies p
LEFT JOIN customers c
    ON p.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


/* ============================================================
   5. POLICY -> AGENT RELATIONSHIP CHECK
   Expected Result: 0 rows
   ============================================================ */

SELECT
    p.policy_id,
    p.agent_id
FROM policies p
LEFT JOIN agents a
    ON p.agent_id = a.agent_id
WHERE a.agent_id IS NULL
  AND p.agent_id IS NOT NULL;


/* ============================================================
   6. CLAIM -> POLICY RELATIONSHIP CHECK
   Expected Result: 0 rows
   ============================================================ */

SELECT
    c.claim_id,
    c.policy_id
FROM claims c
LEFT JOIN policies p
    ON c.policy_id = p.policy_id
WHERE p.policy_id IS NULL;


/* ============================================================
   7. CLAIM -> CUSTOMER RELATIONSHIP CHECK
   ============================================================ */

SELECT
    cl.claim_id,
    cl.customer_id
FROM claims cl
LEFT JOIN customers c
    ON cl.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


/* ============================================================
   8. PREMIUM -> POLICY RELATIONSHIP CHECK
   ============================================================ */

SELECT
    pr.premium_txn_id,
    pr.policy_id
FROM premiums pr
LEFT JOIN policies p
    ON pr.policy_id = p.policy_id
WHERE p.policy_id IS NULL;


/* ============================================================
   9. PREMIUM -> CUSTOMER RELATIONSHIP CHECK
   ============================================================ */

SELECT
    pr.premium_txn_id,
    pr.customer_id
FROM premiums pr
LEFT JOIN customers c
    ON pr.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


/* ============================================================
   10. RENEWAL -> POLICY RELATIONSHIP CHECK
   ============================================================ */

SELECT
    r.renewal_id,
    r.policy_id
FROM renewals r
LEFT JOIN policies p
    ON r.policy_id = p.policy_id
WHERE p.policy_id IS NULL;


/* ============================================================
   11. RENEWAL -> CUSTOMER RELATIONSHIP CHECK
   ============================================================ */

SELECT
    r.renewal_id,
    r.customer_id
FROM renewals r
LEFT JOIN customers c
    ON r.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


/* ============================================================
   12. UNDERWRITING -> POLICY RELATIONSHIP CHECK
   ============================================================ */

SELECT
    u.underwriting_id,
    u.policy_id
FROM underwriting u
LEFT JOIN policies p
    ON u.policy_id = p.policy_id
WHERE p.policy_id IS NULL;


/* ============================================================
   13. UNDERWRITING -> CUSTOMER RELATIONSHIP CHECK
   ============================================================ */

SELECT
    u.underwriting_id,
    u.customer_id
FROM underwriting u
LEFT JOIN customers c
    ON u.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


/* ============================================================
   14. NUMBER OF POLICIES PER CUSTOMER
   ============================================================ */

SELECT
    customer_id,
    COUNT(*) AS total_policies
FROM policies
GROUP BY customer_id
ORDER BY total_policies DESC;


/* ============================================================
   15. NUMBER OF CLAIMS PER POLICY
   ============================================================ */

SELECT
    policy_id,
    COUNT(*) AS total_claims
FROM claims
GROUP BY policy_id
ORDER BY total_claims DESC;


/* ============================================================
   16. NUMBER OF PREMIUM TRANSACTIONS PER POLICY
   ============================================================ */

SELECT
    policy_id,
    COUNT(*) AS total_premium_transactions
FROM premiums
GROUP BY policy_id
ORDER BY total_premium_transactions DESC;