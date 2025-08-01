import json
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
from faker import Faker

fake = Faker()

# Constants
CURRENCIES = ["INR", "USD", "EUR", "GBP"]
CATEGORIES = ["dining", "groceries", "electronics", "travel", "entertainment", "utilities", "healthcare", "education", "investment", "shopping"]
PAYMENT_METHODS = ["credit card", "debit card", "UPI", "net banking", "cash", "wallet"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
MERCHANTS = {
    "dining": ["Zomato", "Swiggy", "FoodPanda", "Uber Eats", "Domino's", "Pizza Hut", "McDonald's", "KFC"],
    "groceries": ["BigBasket", "Grofers", "Amazon Fresh", "Reliance Fresh", "D-Mart", "More", "Spencer's"],
    "electronics": ["Amazon", "Flipkart", "Croma", "Reliance Digital", "Vijay Sales", "Poorvika"],
    "travel": ["IRCTC", "MakeMyTrip", "Goibibo", "Yatra", "Cleartrip", "Airbnb", "Booking.com"],
    "entertainment": ["Netflix", "Amazon Prime", "Hotstar", "BookMyShow", "PVR", "INOX"],
    "utilities": ["BSES", "Tata Power", "Airtel", "Jio", "Vodafone", "Idea", "MTNL"],
    "healthcare": ["Apollo", "Fortis", "Max", "MedPlus", "1mg", "Netmeds", "Pharmeasy"],
    "education": ["Coursera", "Udemy", "edX", "Unacademy", "Byju's", "WhiteHat Jr"],
    "investment": ["Zerodha", "Upstox", "Groww", "Paytm Money", "ICICI Direct", "HDFC Securities"],
    "shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "Purplle", "FirstCry"]
}

def generate_user_details(num_users):
    users = {}
    for i in range(num_users):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = f"{first_name.lower()}.{last_name.lower()}@example.com"
        users[email] = "password123"  # Using a default password for all users
    return users

def generate_transaction(user_id, start_date, end_date):
    category = random.choice(CATEGORIES)
    merchant = random.choice(MERCHANTS[category])
    amount = round(random.uniform(100, 10000), 2)
    timestamp = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "amount": amount,
        "currency": random.choice(CURRENCIES),
        "category": category,
        "merchant_name": merchant,
        "payment_method": random.choice(PAYMENT_METHODS),
        "location": {
            "geocoordinates": [
                round(random.uniform(8.0, 37.0), 6),  # Latitude for India
                round(random.uniform(68.0, 97.0), 6)   # Longitude for India
            ],
            "city": random.choice(CITIES),
            "country": "India"
        },
        "tags": random.sample(["personal", "business", "recurring", "urgent", "gift"], k=random.randint(0, 3)),
        "recurrence": {
            "is_recurring": random.random() < 0.2,
            "frequency": random.choice(["daily", "weekly", "monthly", "yearly"]) if random.random() < 0.2 else ""
        },
        "metadata": {
            "invoice_details": f"INV-{random.randint(1000, 9999)}",
            "itemized_breakdown": [
                {
                    "product": f"Item {chr(65 + i)}",
                    "quantity": random.randint(1, 5),
                    "price": round(amount / (i + 1), 2)
                } for i in range(random.randint(1, 3))
            ],
            "loyalty_points_earned": random.randint(10, 100)
        },
        "keywords": ["transaction", "finance", "payment"],
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat(),
        "expiry_date": "",
        "compatible_user_profiles": random.sample(["frequent_shopper", "tech_savvy", "budget_conscious", "luxury_shopper"], k=random.randint(1, 3)),
        "prerequisites": []
    }

def generate_financial_asset(user_id=None):
    asset_types = ["mutual_fund", "FD", "bond", "equity", "ETF", "NPS", "PPF", "SIP", "401k", "IRA", "Roth IRA", "REIT"]
    asset_type = random.choice(asset_types)
    
    # Define global financial institutions
    global_institutions = {
        "India": [
            "HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank",
            "Reliance Mutual Fund", "HDFC Mutual Fund", "ICICI Prudential", "SBI Mutual Fund",
            "NSE", "BSE", "LIC", "SBI Life", "HDFC Life"
        ],
        "USA": [
            "JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs",
            "Morgan Stanley", "Vanguard", "Fidelity", "BlackRock", "Charles Schwab"
        ],
        "UK": [
            "HSBC", "Barclays", "Lloyds Bank", "Royal Bank of Scotland",
            "Standard Chartered", "Aviva", "Legal & General", "Prudential"
        ],
        "Singapore": [
            "DBS Bank", "OCBC Bank", "UOB", "Temasek Holdings",
            "GIC", "Singtel", "Keppel Corporation"
        ]
    }
    
    # Select a random country
    country = random.choice(list(global_institutions.keys()))
    institution = random.choice(global_institutions[country])
    
    # Define country-specific tax implications
    tax_implications = {
        "India": {
            "short_term": f"{random.randint(10, 30)}%",
            "long_term": f"{random.randint(5, 20)}% after 1 year",
            "tax_saving": random.choice(["80C eligible", "80D eligible", "Not eligible"])
        },
        "USA": {
            "short_term": f"{random.randint(10, 37)}%",
            "long_term": f"{random.randint(0, 20)}% after 1 year",
            "tax_saving": random.choice(["401k eligible", "IRA eligible", "Roth IRA eligible", "Not eligible"])
        },
        "UK": {
            "short_term": f"{random.randint(20, 45)}%",
            "long_term": f"{random.randint(10, 20)}% after 1 year",
            "tax_saving": random.choice(["ISA eligible", "Pension eligible", "Not eligible"])
        },
        "Singapore": {
            "short_term": f"{random.randint(0, 22)}%",
            "long_term": f"{random.randint(0, 22)}% after 1 year",
            "tax_saving": random.choice(["CPF eligible", "SRS eligible", "Not eligible"])
        }
    }
    
    # Define country-specific prerequisites
    prerequisites = {
        "India": ["Demat account", "KYC", "Bank account", "PAN card", "Aadhaar card"],
        "USA": ["Social Security Number", "Bank account", "Brokerage account"],
        "UK": ["National Insurance Number", "Bank account", "Brokerage account"],
        "Singapore": ["NRIC", "Bank account", "CDP account"]
    }
    
    return {
        "asset_id": str(uuid.uuid4()),
        "user_id": user_id,
        "type": asset_type,
        "name": f"{institution} {asset_type.replace('_', ' ').title()}",
        "issuer": institution,
        "country": country,
        "risk_rating": random.randint(1, 5),
        "expected_return": round(random.uniform(5, 15), 2),
        "liquidity": random.choice(["low", "medium", "high"]),
        "minimum_investment_amount": random.choice([1000, 5000, 10000, 50000, 100000]),
        "tenure": f"{random.randint(1, 10)} years",
        "financial_details": {
            "historical_performance": {
                str(year): f"{round(random.uniform(5, 20), 1)}%" 
                for year in range(2020, datetime.now().year)
            },
            "tax_implications": tax_implications[country],
            "key_features": random.sample([
                "dividend-paying", "tax-saving", "growth-oriented", 
                "income-focused", "index-tracking", "sector-specific"
            ], k=random.randint(2, 4))
        },
        "metadata": {
            "regulatory_documents": [f"https://example.com/prospectus_{random.randint(1000, 9999)}.pdf"],
            "tags": random.sample(["ESG", "high-growth", "stable", "aggressive", "conservative"], k=random.randint(1, 3))
        },
        "keywords": ["investment", asset_type, "finance", country],
        "created_at": fake.date_time_this_year().isoformat(),
        "updated_at": fake.date_time_this_year().isoformat(),
        "expiry_date": "",
        "compatible_user_profiles": random.sample(["investors", "retirees", "young_professionals", "risk_averse"], k=random.randint(1, 3)),
        "prerequisites": random.sample(prerequisites[country], k=random.randint(1, 3))
    }

def generate_investment_strategy():
    # Define Indian investment strategy types
    strategy_types = [
        "Retirement", "Growth", "Income", "Balanced", "Tax Saving",
        "NPS Focused", "PPF Based", "ELSS Based", "Hybrid"
    ]
    
    # Define Indian asset types with descriptions
    indian_asset_types = {
        "FD": "Fixed Deposits",
        "mutual_funds": "Mutual Funds",
        "bonds": "Government Bonds",
        "equity": "Equity",
        "gold": "Gold",
        "NPS": "National Pension Scheme",
        "PPF": "Public Provident Fund",
        "ELSS": "Equity Linked Savings Scheme",
        "SIP": "Systematic Investment Plan"
    }
    
    # Define Indian financial institutions
    indian_institutions = [
        "HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank",
        "Reliance Mutual Fund", "HDFC Mutual Fund", "ICICI Prudential", "SBI Mutual Fund",
        "NSE", "BSE", "LIC", "SBI Life", "HDFC Life"
    ]
    
    return {
        "strategy_id": str(uuid.uuid4()),
        "name": f"{random.choice(indian_institutions)} {random.choice(strategy_types)} Plan",
        "risk_profile": random.choice(["conservative", "moderate", "aggressive"]),
        "time_horizon": random.choice(["short-term", "medium-term", "long-term"]),
        "target_annual_return": round(random.uniform(5, 15), 1),
        "allocation_blueprint": {
            asset_type: random.randint(10, 40)
            for asset_type in indian_asset_types.keys()
        },
        "performance_metrics": {
            "backtested_results": f"{round(random.uniform(5, 15), 1)}% CAGR over {random.randint(5, 15)} years",
            "volatility_score": round(random.uniform(1, 5), 1),
            "tax_efficiency_rating": random.choice(["low", "medium", "high"])
        },
        "user_requirements": {
            "minimum_capital": random.choice([10000, 50000, 100000, 500000, 1000000]),
            "recommended_account_types": random.sample(["Demat", "Savings", "NPS", "PPF", "ELSS"], k=random.randint(1, 3))
        },
        "keywords": ["investment", "strategy", "portfolio", "India"],
        "created_at": fake.date_time_this_year().isoformat(),
        "updated_at": fake.date_time_this_year().isoformat(),
        "expiry_date": "",
        "compatible_user_profiles": random.sample(["retirees", "risk_averse", "young_professionals", "high_net_worth"], k=random.randint(1, 3)),
        "prerequisites": random.sample(["KYC", "Demat account", "Bank account", "PAN card", "Aadhaar card"], k=random.randint(1, 3))
    }

def generate_dataset(num_users=100, transactions_per_user=50, assets_per_user=5, num_strategies=20):
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    # Generate user details
    users = generate_user_details(num_users)
    
    # Generate transactions and assets for each user
    transactions = []
    financial_assets = []
    
    for user_id in users.keys():
        # Generate transactions
        transactions.extend([
            generate_transaction(user_id, start_date, end_date)
            for _ in range(transactions_per_user)
        ])
        
        # Generate assets for this user
        financial_assets.extend([
            generate_financial_asset(user_id)
            for _ in range(assets_per_user)
        ])
    
    # Generate investment strategies (not user-specific)
    investment_strategies = [
        generate_investment_strategy()
        for _ in range(num_strategies)
    ]
    
    return {
        "users": users,
        "transactions": transactions,
        "financial_assets": financial_assets,
        "investment_strategies": investment_strategies
    }

if __name__ == "__main__":
    import os
    os.makedirs("backend", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("fraud_detection", exist_ok=True)
    # Generate a large dataset
    dataset = generate_dataset(
        num_users=10,          # 1000 users
        transactions_per_user=20,  # 100 transactions per user
        assets_per_user=15,       # 5 assets per user
        num_strategies=1000        # 50 investment strategies
    )

    # Save users to users.json
    with open("backend/users.json", "w") as f:
        json.dump(dataset["users"], f, indent=2)

    # Save transactions in the format required by fraud_detection/transactions.json
    # Convert merchant_name to merchant, and location to city string
    transactions_for_fraud = []
    for txn in dataset["transactions"]:
        transactions_for_fraud.append({
            "user_id": txn["user_id"],
            "amount": txn["amount"],
            "merchant": txn["merchant_name"],
            "location": txn["location"]["city"] if isinstance(txn["location"], dict) else txn["location"],
            "timestamp": txn["timestamp"],
            "category": txn["category"]
        })
    with open("fraud_detection/transactions.json", "w") as f:
        json.dump(transactions_for_fraud, f, indent=2)

    # Generate fraud_vectors.json using a simple rule-based approach
    major_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
    fraud_vectors = []
    for txn in transactions_for_fraud:
        if txn["amount"] > 5000 or txn["location"] not in major_cities:
            fraud_vectors.append(txn)
    with open("fraud_detection/fraud_vectors.json", "w") as f:
        json.dump(fraud_vectors, f, indent=2)

    # Save other data to large_financial_data.json
    financial_data = {
        "transactions": dataset["transactions"],
        "financial_assets": dataset["financial_assets"],
        "investment_strategies": dataset["investment_strategies"]
    }
    with open("data/large_financial_data.json", "w") as f:
        json.dump(financial_data, f, indent=2)

    print(f"Generated dataset with:")
    print(f"- {len(dataset['users'])} users")
    print(f"- {len(dataset['transactions'])} transactions")
    print(f"- {len(dataset['financial_assets'])} financial assets")
    print(f"- {len(dataset['investment_strategies'])} investment strategies") 