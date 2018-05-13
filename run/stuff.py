import numpy as np
from sklearn.preprocessing import OneHotEncoder

from base.card import Card, CardSuit, CardValue
from base.deck import Deck

if __name__ == '__main__':
    encoder = OneHotEncoder()
    encoder.fit(Deck().cards)
    print(encoder.transform([Card(CardValue.THREE, CardSuit.HEARTS)]))