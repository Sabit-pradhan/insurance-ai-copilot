/* ============================================================
   INSURANCE AI COPILOT PROJECT
   FILE: 02_business_queries.sql

   PURPOSE:
   Business analysis queries for:
   1. Customers
   2. Policies
   3. Claims
   4. Premiums
   5. Agents
   6. Renewals
   ============================================================ */


/* ============================================================
   QUERY 1
   Total Customers
   ============================================================ */

SELECT COUNT(*) AS total_customers
FROM customers;



/* ============================================================
   QUERY 2
   Total Policies
   ============================================================ */

SELECT COUNT(*) AS total_policies
FROM policies;



/* ============================================================
   QUERY 3
   Policies per Customer
   ============================================================ */

SELECT
    customer_id,
    COUNT(*) AS total_policies
FROM policies
GROUP BY customer_id
ORDER BY total_policies DESC;



/* ============================================================
   QUERY 4
   Total Claims
   ============================================================ */

SELECT COUNT(*) AS total_claims
FROM claims;



/* ============================================================
   QUERY 5
   Claims per Policy
   ============================================================ */

SELECT
    policy_id,
    COUNT(*) AS total_claims
FROM claims
GROUP BY policy_id
ORDER BY total_claims DESC;



/* ============================================================
   QUERY 6
   Customer + Policy Details
   ============================================================ */

SELECT
    c.customer_id,
    c.customer_name,
    p.policy_id,
    p.policy_type,
    p.policy_status
FROM customers c
JOIN policies p
    ON c.customer_id = p.customer_id;



/* ============================================================
   QUERY 7
   Policies Sold by Each Agent
   ============================================================ */

SELECT
    agent_id,
    COUNT(*) AS policies_sold
FROM policies
GROUP BY agent_id
ORDER BY policies_sold DESC;



/* ============================================================
   QUERY 8
   Premium Transactions per Policy
   ============================================================ */

SELECT
    policy_id,
    COUNT(*) AS total_transactions
FROM premium_transactions
GROUP BY policy_id
ORDER BY total_transactions DESC;



/* ============================================================
   QUERY 9
   Policies with Claims
   ============================================================ */

SELECT DISTINCT
    p.policy_id,
    p.customer_id,
    p.policy_type
FROM policies p
JOIN claims c
    ON p.policy_id = c.policy_id;



/* ============================================================
   QUERY 10
   Customers who have made Claims
   ============================================================ */

SELECT DISTINCT
    cu.customer_id,
    cu.customer_name
FROM customers cu
JOIN policies p
    ON cu.customer_id = p.customer_id
JOIN claims c
    ON p.policy_id = c.policy_id;



/* ============================================================
   QUERY 11
   Claim Count by Policy Type
   ============================================================ */

SELECT
    p.policy_type,
    COUNT(c.claim_id) AS total_claims
FROM policies p
LEFT JOIN claims c
    ON p.policy_id = c.policy_id
GROUP BY p.policy_type
ORDER BY total_claims DESC;



/* ============================================================
   QUERY 12
   Policies Without Claims
   ============================================================ */

SELECT
    p.policy_id,
    p.customer_id,
    p.policy_type
FROM policies p
LEFT JOIN claims c
    ON p.policy_id = c.policy_id
WHERE c.claim_id IS NULL;



/* ============================================================
   QUERY 13
   Renewal Count
   ============================================================ */

SELECT COUNT(*) AS total_renewal_records
FROM renewals;



/* ============================================================
   QUERY 14
   Policy + Renewal Details
   ============================================================ */

SELECT
    p.policy_id,
    p.customer_id,
    p.policy_type,
    r.*
FROM policies p
LEFT JOIN renewals r
    ON p.policy_id = r.policy_id;



/* ============================================================
   QUERY 15
   Policy + Underwriting Details
   ============================================================ */

SELECT
    p.policy_id,
    p.customer_id,
    p.policy_type,
    u.*
FROM policies p
LEFT JOIN underwriting u
    ON p.policy_id = u.policy_id;