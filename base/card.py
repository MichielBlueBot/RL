from enum import Enum


class CardSuit(Enum):

    HEARTS = "♥"
    DIAMONDS = "♦"
    SPADES = "♠"
    CLUBS = "♣"


class CardValue(Enum):

    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    JOKER = "$"


class Card:

    def __init__(self,  value, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return "{}{}".format(self.suit.value, self.value.value)
