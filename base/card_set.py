from typing import List

from sklearn.preprocessing import OneHotEncoder

from base.deck import Deck


class CardSet(List):

    def __init__(self, opener="[", closer="]"):
        super().__init__()
        self.opener = opener
        self.closer = closer
        self.encoder = self._create_card_set_encoder()

    def get_state(self):
        return self.encoder.transform(self)

    @staticmethod
    def _create_card_set_encoder():
        encoder = OneHotEncoder()
        encoder.fit(Deck().cards)
        return encoder

    def __repr__(self):
        result = self.opener
        for card in self:
            result += str(card) + ", "
        if len(result) > 1:
            result = result[:-2]
        result += self.closer
        return result
