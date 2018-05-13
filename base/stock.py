from base.card import Card
from base.card_set import CardSet


class Stock:
    def __init__(self):
        self.cards = CardSet()

    def add(self, card: Card):
        self.cards.append(card)

    def take(self):
        cards = [card for card in self.cards]
        self.cards = []
        return cards

    def __repr__(self):
        return "Stock - {}".format(self.cards)
