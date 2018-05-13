from random import shuffle

from base.card import CardSuit, CardValue, Card
from base.card_set import CardSet


class Deck:

    def __init__(self, shuffled=True):
        self.cards = CardSet()
        self.reset()
        if shuffled:
            self.shuffle()

    def pop(self):
        if self.cards:
            return self.cards.pop(0)
        return None

    def pop_n(self, n=1):
        cards = []
        for _ in range(n):
            new_card = self.pop()
            if new_card:
                cards.append(new_card)
            else:
                break
        return cards

    def size(self):
        return len(self.cards)

    def empty(self):
        return len(self.cards) == 0

    def shuffle(self):
        shuffle(self.cards)

    def reset(self):
        for card_value in CardValue:
            for card_suit in CardSuit:
                self.cards.append(Card(card_value, card_suit))

    def get_state(self):
        return self.cards.get_state()

    def __repr__(self):
        return "Deck - {}".format(self.cards)

