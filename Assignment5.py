
from games import Game, play_game
from games import query_player, random_player, alphabeta_full_player,alphabeta_player
from utils import Dict, Struct, update, if_, num_or_str

#_______________________________________________
# Tic Tac Toe

#Player for tic_tac_toe, more robust than default query player
def tic_tac_query_player(game, state):
    "Make a move by querying standard input."
    tryAgain = True
    while tryAgain:
	    game.display(state)
	    val = num_or_str(input('Your move, in format x,y (ranging from 1-3): '))
	    tmp = tuple(int(item) for item in val.split(',') if item.strip())
	    if ((tmp[0]>=1 and tmp[0]<=3) and (tmp[1]>=1 and tmp[1]<=3)):
	    	if tmp in state.moves:
	    		tryAgain=False
	    	else:
	    		print("Illegal move, try again")
	    else:
	    	print("Input(s) out of bounds, try again")
    return tmp
    
class TicTacToe(Game):
    """Play TicTacToe on an h x v board with k in a line required to win, 
    and Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""
    def __init__(self, h=3, v=3, k=3):
        update(self, h=h, v=v, k=k)
        moves = [(x, y) for x in range(1, h+1)
                 for y in range(1, v+1)]
        self.initial = Struct(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        "Legal moves are any square not yet taken."
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            print("Illegal Move")
            return state # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return Struct(to_move=if_(state.to_move == 'X', 'O', 'X'),
                      utility=self.compute_utility(board, move, state.to_move),
                      board=board, moves=moves)

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        return if_(player == 'X', state.utility, -state.utility)

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h+1):
            for y in range(1, self.v+1):
                print (board.get((x, y), '.'),end=" ")
            print()

    def compute_utility(self, board, move, player):
        "If X wins with this move, return 1; if O, return -1; else return 0."
        if (self.k_in_row(board, move, player, (0, 1)) or
            self.k_in_row(board, move, player, (1, 0)) or
            self.k_in_row(board, move, player, (1, -1)) or
            self.k_in_row(board, move, player, (1, 1))):
            return if_(player == 'X', +1, -1)
        else:
            return 0

    def k_in_row(self, board, move, player, delta):
        "Return true if there is a line through move on board for player."
        delta_x, delta_y = delta[0],delta[1]
        x, y = move
        n = 0 # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1 # Because we counted move itself twice
        return n >= self.k

#____________________________________________________________
#
print(play_game(TicTacToe(), random_player, random_player)) 

"""
print(play_game(TicTacToe(), tic_tac_query_player,random_player))
print(play_game(TicTacToe(),tic_tac_query_player,alphabeta_player))
print(play_game(TicTacToe(), random_player, random_player)) 
print(play_game(TicTacToe(), alphabeta_player,alphabeta_player)) 
"""

