"""Step 6 - Explore the cleaned churn dataset with SQLite queries."""

from pathlib import Path
import sqlite3

import pandas as pd

from config import DATA_CLEANED, TARGET


TABLE_NAME = "customers"


def run_query(connection, query):
    """Execute a SQL query and return its result as a DataFrame."""
    return pd.read_sql_query(query, connection)


def print_result(label, result):
    """Print a consistently formatted query result."""
    print(f"\n{label}")
    print("-" * len(label))
    print(result.to_string(index=False))


def main():
    """Load data into SQLite memory and run the business analysis queries."""
    df = pd.read_csv(DATA_CLEANED)

    with sqlite3.connect(":memory:") as connection:
        df.to_sql(TABLE_NAME, connection, index=False, if_exists="replace")

        print("SQL Queries on Churn Dataset")
        print("=" * 35)
        print("6.1 Setup: SQLite In-Memory")
        print(f"Loaded {len(df):,} rows into the '{TABLE_NAME}' table.")

        # 6.2 Basic Exploration (Q01-Q03)
        print("\n6.2 Basic Exploration")
        q01 = run_query(connection, """
            SELECT COUNT(*) AS total_customers,
                   SUM("Churn") AS churned_customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers;
        """)
        print_result("Q01 - Customer count and churn rate", q01)

        q02 = run_query(connection, """
            SELECT "Churn" AS churn_status,
                   COUNT(*) AS customers,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2)
                       AS percentage
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """)
        print_result("Q02 - Churn distribution", q02)

        q03 = run_query(connection, """
            SELECT MIN("Age") AS minimum_age,
                   ROUND(AVG("Age"), 2) AS average_age,
                   MAX("Age") AS maximum_age,
                   ROUND(AVG("Tenure"), 2) AS average_tenure,
                   ROUND(AVG("Total Spend"), 2) AS average_total_spend
            FROM customers;
        """)
        print_result("Q03 - Overall customer statistics", q03)

        # 6.3 Subscription & Contract Analysis (Q04-Q05)
        print("\n6.3 Subscription & Contract Analysis")
        q04 = run_query(connection, """
            SELECT CASE
                       WHEN "Subscription Type_Premium" = 1 THEN 'Premium'
                       WHEN "Subscription Type_Standard" = 1 THEN 'Standard'
                       ELSE 'Basic'
                   END AS subscription_type,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY subscription_type
            ORDER BY churn_rate_percent DESC;
        """)
        print_result("Q04 - Churn by subscription type", q04)

        q05 = run_query(connection, """
            SELECT CASE
                       WHEN "Contract Length_Monthly" = 1 THEN 'Monthly'
                       WHEN "Contract Length_Quarterly" = 1 THEN 'Quarterly'
                       ELSE 'Annual'
                   END AS contract_length,
                   COUNT(*) AS customers,
                   ROUND(AVG("Total Spend"), 2) AS average_spend,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY contract_length
            ORDER BY churn_rate_percent DESC;
        """)
        print_result("Q05 - Contract performance", q05)

        # 6.4 Spend Analysis (Q06-Q07)
        print("\n6.4 Spend Analysis")
        q06 = run_query(connection, """
            SELECT "Churn" AS churn_status,
                   ROUND(AVG("Total Spend"), 2) AS average_total_spend,
                   ROUND(MIN("Total Spend"), 2) AS minimum_spend,
                   ROUND(MAX("Total Spend"), 2) AS maximum_spend
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """)
        print_result("Q06 - Spend by churn status", q06)

        q07 = run_query(connection, """
            SELECT CASE
                       WHEN "Total Spend" < 500 THEN 'Under 500'
                       WHEN "Total Spend" < 1000 THEN '500 to 999'
                       ELSE '1000 or more'
                   END AS spend_band,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY spend_band
            ORDER BY MIN("Total Spend");
        """)
        print_result("Q07 - Churn by spend band", q07)

        # 6.5 Support & Usage (Q08-Q10)
        print("\n6.5 Support & Usage")
        q08 = run_query(connection, """
            SELECT "Support Calls" AS support_calls,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY "Support Calls"
            ORDER BY "Support Calls";
        """)
        print_result("Q08 - Churn by support calls", q08)

        q09 = run_query(connection, """
            SELECT CASE
                       WHEN "Usage Frequency" < 10 THEN 'Low usage'
                       WHEN "Usage Frequency" < 20 THEN 'Medium usage'
                       ELSE 'High usage'
                   END AS usage_band,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY usage_band
            ORDER BY MIN("Usage Frequency");
        """)
        print_result("Q09 - Churn by usage frequency", q09)

        q10 = run_query(connection, """
            SELECT "Support Calls" AS support_calls,
                   ROUND(AVG("Usage Frequency"), 2) AS average_usage,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY "Support Calls"
            ORDER BY churn_rate_percent DESC;
        """)
        print_result("Q10 - Support calls and average usage", q10)

        # 6.6 Tenure Analysis (Q11-Q12)
        print("\n6.6 Tenure Analysis")
        q11 = run_query(connection, """
            SELECT CASE
                       WHEN "Tenure" < 12 THEN 'Under 12 months'
                       WHEN "Tenure" < 24 THEN '12 to 23 months'
                       ELSE '24 months or more'
                   END AS tenure_band,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY tenure_band
            ORDER BY MIN("Tenure");
        """)
        print_result("Q11 - Churn by tenure band", q11)

        q12 = run_query(connection, """
            SELECT "Churn" AS churn_status,
                   ROUND(AVG("Tenure"), 2) AS average_tenure,
                   ROUND(AVG("Last Interaction"), 2) AS average_last_interaction
            FROM customers
            GROUP BY "Churn"
            ORDER BY "Churn";
        """)
        print_result("Q12 - Tenure and recency by churn status", q12)

        # 6.7 Payment & Demographics (Q13-Q15)
        print("\n6.7 Payment & Demographics")
        q13 = run_query(connection, """
            SELECT CASE
                       WHEN "Payment Delay" < 15 THEN 'Under 15 days'
                       WHEN "Payment Delay" < 30 THEN '15 to 29 days'
                       ELSE '30 days or more'
                   END AS payment_delay_band,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY payment_delay_band
            ORDER BY MIN("Payment Delay");
        """)
        print_result("Q13 - Churn by payment delay", q13)

        q14 = run_query(connection, """
            SELECT "Gender" AS gender,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY "Gender"
            ORDER BY gender;
        """)
        print_result("Q14 - Churn by gender", q14)

        q15 = run_query(connection, """
            SELECT CASE
                       WHEN "Age" < 30 THEN 'Under 30'
                       WHEN "Age" < 50 THEN '30 to 49'
                       ELSE '50 or older'
                   END AS age_band,
                   COUNT(*) AS customers,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY age_band
            ORDER BY MIN("Age");
        """)
        print_result("Q15 - Churn by age band", q15)

        # 6.8 Key Business Insights from SQL
        payment_insight = run_query(connection, """
            SELECT CASE WHEN "Payment Delay" >= 30 THEN '30+ days' ELSE 'Under 30 days' END AS segment,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY segment
            ORDER BY churn_rate_percent DESC
            LIMIT 1;
        """).iloc[0]
        support_insight = run_query(connection, """
            SELECT "Support Calls" AS support_calls,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY "Support Calls"
            ORDER BY churn_rate_percent DESC
            LIMIT 1;
        """).iloc[0]
        contract_insight = run_query(connection, """
            SELECT CASE
                       WHEN "Contract Length_Monthly" = 1 THEN 'Monthly'
                       WHEN "Contract Length_Quarterly" = 1 THEN 'Quarterly'
                       ELSE 'Annual'
                   END AS contract_length,
                   ROUND(100.0 * AVG("Churn"), 2) AS churn_rate_percent
            FROM customers
            GROUP BY contract_length
            ORDER BY churn_rate_percent DESC
            LIMIT 1;
        """).iloc[0]

        print("\n6.8 Key Business Insights from SQL")
        print("- Customers with the highest payment-delay segment show the greatest churn risk.")
        print(f"  Highest payment-delay segment: {payment_insight['segment']} ({payment_insight['churn_rate_percent']:.2f}% churn).")
        print(f"- The highest observed support-call group is {int(support_insight['support_calls'])} calls ({support_insight['churn_rate_percent']:.2f}% churn).")
        print(f"- The contract type with the highest churn is {contract_insight['contract_length']} ({contract_insight['churn_rate_percent']:.2f}% churn).")
        print("- These segments are useful targets for payment reminders, support follow-up, and retention offers.")


if __name__ == "__main__":
    main()
