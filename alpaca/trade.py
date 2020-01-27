# ALgo trading test script for paper market with Alpaca

import requests
from alpaca import config
import json

BASE_URL = 'https://paper-api.alpaca.markets'
ACCOUNT_URL = f'{BASE_URL}/v2/account'
ORDERS_URL = f'{BASE_URL}/v2/orders'
AUTH_HEADERS = {'APCA-API-KEY-ID': config.PAPER_API_KEY_ID, 'APCA-API-SECRET-KEY': config.PAPER_SECRET_KEY}


def get_account():
    req = requests.get(ACCOUNT_URL, headers=AUTH_HEADERS)

    return json.loads(req.content)


def place_order(symbol, qty, side, type, time_in_force):
    data = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': type,
        'time_in_force': time_in_force,
    }
    req = requests.post(ORDERS_URL, headers=AUTH_HEADERS, json=data)
    return json.loads(req.content)



if __name__=='__main__':
    response = place_order('TSLA', 400, 'buy', 'market', 'day')
    print(response)

print('Praise the Sun')
