import random

from base.card_set import CardSet
from base.deck import Deck
from base.stock import Stock


class Player:

    def __init__(self):
        self.hand = CardSet()

    def hand_size(self):
        return len(self.hand)

    def pop_from_deck(self, deck: Deck, amount=1):
        cards = deck.pop_n(amount)
        if cards:
            self.hand.extend(cards)
        else:
            raise Exception("Player - Tried to pop from empty deck!")

    def take_stock(self, stock: Stock):
        self.hand.extend(stock.take())

    def throw_to_stock(self, stock: Stock):
        # TODO: add non-random throw-away
        card_idx = random.randint(0, self.hand_size() - 1)
        stock.add(self.hand[card_idx])
        self.hand.remove(self.hand[card_idx])

    def perform_action(self, deck: Deck, stock: Stock):
        # Decide whether to take the stock or not
        # TODO: add actions
        if random.random() < 0.5:
            # Take the stock
            self.take_stock(stock)
        else:
            # Take from deck
            self.pop_from_deck(deck)
        # Throw away a card to the stock
        self.throw_to_stock(stock)

    def get_state(self):
        return self.hand.get_state()

    def __repr__(self):
        return "Player - {}".format(self.hand)

