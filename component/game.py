from typing import List

from base.deck import Deck
from base.stock import Stock
from component.player import Player


class Game:

    def __init__(self, num_players=4):
        self.num_players = num_players
        self.player_turn_index = 0
        self.players = []  # type: List[Player]
        self.deck = None  # type: Deck
        self.stock = None  # type: Stock
        self.initialize()

    def initialize(self):
        # Create players
        for _ in range(self.num_players):
            self.players.append(Player())
        # Give each player a 13-card hand
        deck = Deck()
        while not deck.empty():
            for player in self.players:
                player.pop_from_deck(deck)
        # Initialize deck and stock
        self.deck = Deck()
        self.stock = Stock()

    def is_episode_finished(self):
        return self.deck.empty()

    @staticmethod
    def new_episode():
        print("Starting new episode ...")

    def get_state(self) -> List[float]:
        state = []
        state.extend(self.deck.get_state())
        state.extend(self.players[0].get_state())
        return state

    def perform_action(self, policy):
        pass

    def play_step(self):
        current_player = self.players[self.player_turn_index]
        current_player.perform_action(self.deck, self.stock)
        self.player_turn_index += 1
        self.player_turn_index %= len(self.players)

    def print(self):
        print("============== GAME ===============")
        print(self.deck)
        print(self.stock)
        for player in self.players:
            print(player)

