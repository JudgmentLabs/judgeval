financial_data = [{
    "information": """
    Unrealized PnL shows how your position is performing at a snapshot in time, but no transaction has taken place to lock in gains or losses.
    
    Unrealized PnL = (Current Market Price – Purchase Price) × Number of Shares

    Purchase Price is the price at which you originally bought the shares.
    Current Market Price is the price of the stock at the time of calculation.
    Number of Shares is how many shares you hold.
    
    Example of Unrealized PnL

    You bought 100 shares of XYZ at $50 per share.
    The current market price is $55 per share.
    
    Unrealized PnL = (55 – 50) × 100 = 5 × 100 = $500
    
    **IMPORTANT NOTE:** It's okay to use whatever prices you want to do these calculations.
    """,
    "category": "pnl"
},
    {
        "information": """
        Table schema:
        ### Table Schema Definition

        Let's assume we have a table named `stock_transactions` with the following columns:

        - `id` (INT): A unique identifier for each transaction.
        - `stock_symbol` (VARCHAR): The ticker symbol of the stock (e.g., AAPL for Apple).
        - `transaction_type` (VARCHAR): Type of transaction, either 'buy' or 'sell'.
        - `quantity` (INT): The number of shares bought or sold.
        - `price_per_share` (DECIMAL): The price per share at the time of the transaction.
        - `transaction_date` (DATE): The date when the transaction occurred.

        ```sql
        CREATE TABLE stock_transactions (
            id INT PRIMARY KEY,
            stock_symbol VARCHAR(10),
            transaction_type VARCHAR(4),
            quantity INT,
            price_per_share DECIMAL(10, 2),
            transaction_date DATE
        );
        ```
        """,
        "category": "pnl"
    },
    {
        "information": """
        stock_transactions table:
        aapl buy 100 100 2024-01-01
        aapl sell 50 150 2024-01-02
        """,
        "category": "pnl"
    },
    {
        "information": """
        Common ticker symbols:
        Tesla: TSLA
        Apple: AAPL
        Microsoft: MSFT
        Amazon: AMZN
        Google: GOOGL
        Facebook: META
        Netflix: NFLX
        """,
        "category": "stocks"
    },
    {
        "information": """
        Table schema:
        ### Table Schema Definition

        Let's assume we have a table named `stock_transactions` with the following columns:

        - `id` (INT): A unique identifier for each transaction.
        - `stock_symbol` (VARCHAR): The ticker symbol of the stock (e.g., AAPL for Apple).
        - `transaction_type` (VARCHAR): Type of transaction, either 'buy' or 'sell'.
        - `quantity` (INT): The number of shares bought or sold.
        - `price_per_share` (DECIMAL): The price per share at the time of the transaction.
        - `transaction_date` (DATE): The date when the transaction occurred.

        ```sql
        CREATE TABLE stock_transactions (
            id INT PRIMARY KEY,
            stock_symbol VARCHAR(10),
            transaction_type VARCHAR(4),
            quantity INT,
            price_per_share DECIMAL(10, 2),
            transaction_date DATE
        );
        ```
        """,
        "category": "stocks"
    }    
]

incorrect_financial_data = [{
    "information": """
    Unrealized PnL shows how your position is performing at a snapshot in time, but no transaction has taken place to lock in gains or losses.

    Unrealized PnL = Current Market Price – Purchase Price

    Purchase Price is the price at which you originally bought the shares.
    Current Market Price is the price of the stock at the time of calculation.
    Number of Shares is how many shares you hold.
    
    Example of Unrealized PnL

    You bought 100 shares of XYZ at $50 per share.
    The current market price is $55 per share.
    
    Unrealized PnL = 55 – 50 = $5
    """,
    "category": "pnl"
}, 
{
        "information": """
        Table schema:
        ### Table Schema Definition

        Let's assume we have a table named `stock_transactions` with the following columns:

        - `id` (INT): A unique identifier for each transaction.
        - `stock_symbol` (VARCHAR): The ticker symbol of the stock (e.g., appl for Apple).
        - `transaction_type` (VARCHAR): Type of transaction, either 'buy' or 'sell'.
        - `quantity` (INT): The number of shares bought or sold.
        - `price_per_share` (DECIMAL): The price per share at the time of the transaction.
        - `transaction_date` (DATE): The date when the transaction occurred.

        ```sql
        CREATE TABLE stock_transactions (
            id INT PRIMARY KEY,
            stock_symbol VARCHAR(10),
            transaction_type VARCHAR(4),
            quantity INT,
            price_per_share DECIMAL(10, 2),
            transaction_date DATE
        );
        ```
        """,
        "category": "pnl"
    },
    {
        "information": """
        stock_transactions table:
        appl buy 100 100 2024-01-01
        appl sell 50 150 2024-01-02
        """,
        "category": "pnl"
    },
    {
        "information": """
        Common ticker symbols:
        Tesla: TSSL
        Apple: APPL
        Microsoft: MCST
        Amazon: AMZN.A
        Google: GOOG.L
        Facebook: META.X
        Netflix: NFLX.B
        """,
        "category": "stocks"
    },
        {
        "information": """
        stock_transactions table:
        appl buy 100 100 2024-01-01
        appl sell 50 150 2024-01-02
        """,
        "category": "stocks"
    },
]