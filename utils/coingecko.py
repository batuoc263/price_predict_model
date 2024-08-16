import requests

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=90"
        return url
    else:
        raise ValueError("Unsupported token")
    
def get_coingecko_data(token):
    url = get_coingecko_url(token)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        data = {}

    return data