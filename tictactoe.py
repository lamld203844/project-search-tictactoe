"""
Tic Tac Toe Player
"""

import math
import copy
import numpy as np

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def empty_count(board):
    empty_num = 0
    for entry in board:
        empty_num += entry.count(EMPTY)
    return empty_num

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    empty_num = empty_count(board)

    # return None if the game is over
    if empty_num == 0 or terminal(board):
        winner(board)
        return None
    elif empty_num % 2 == 0:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # case: game is over
    if terminal(board):
        winner(board)
        return
    else:
        # return a set of all of the tuple possible actions 
        possible_actions = set()

        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == EMPTY:
                    possible_actions.add((i,j))

        return possible_actions
                    
def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # check valid action
    if (not (0 <= action[0] <= 2)) or (not (0 <= action[1] <= 2)):
        raise Exception('Invalid action')

    # deep copy
    result_board = copy.deepcopy(board)

    result_board[action[0]][action[1]] = player(board)
    
    return result_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    def check_rows(matrix):
        list = []
        # loop row over matrix
        for row in matrix:
            # appending if all entry in row - same type
            if len(set(row)) == 1:
                list.append(row[0])

        # At most one winner condition
        return list[0] if len(set(list)) == 1 else None
    
    def check_diagonals(matrix):
        len_matrix = len(matrix)
        main = []
        transpose = []
        # get main, transpose diagonal respectively
        for i in range(len_matrix):
            main.append(matrix[i][i])
            transpose.append(matrix[i][len_matrix-i-1])

        # return if all entry in list - same type
        if len(set(main)) == 1:
            return main[0]
        elif len(set(transpose)) == 1:
            return transpose[0]

    def check_cols(matrix):
        # unmodifing original board
        matrix_copy = copy.deepcopy(matrix)
        # transposing
        matrix_copy = np.array(matrix_copy).T.tolist()
        return check_rows(matrix_copy)

    # check rows
    if check_rows(board) is not None:
        return check_rows(board)
    # check diagonals
    elif check_diagonals(board) is not None:
        return check_diagonals(board)
    # check cols = check rows(transposing board)
    elif check_cols(board) is not None:
        return check_cols(board)


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    empty_counter = 0
    for row in board:
        empty_counter += row.count(EMPTY)
    
    return True if (empty_counter == 0) or (winner(board) is not None) else False

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # only be called if game is over
    if terminal(board):
        if winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
        else:
            return 0
    
    return


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Return random movement if initializing
    if empty_count(board) == 9:
        return (1,1)
    else:
        def max_value(state):
            # check if the game is over
            if terminal(state):
                return utility(state)
            
            v = -math.inf
            for action in actions(state):
                v = max(v, min_value(result(state, action)))
            
            return v

        def min_value(state):
            # check if the game is over
            if terminal(state):
                return utility(state)
                
            v = math.inf
            for action in actions(state):
                v = min(v, max_value(result(state, action)))
                
            return v

        # Given a board
        #	Maximizer
        if player(board) == X:
            # picks action in actions(board) that produces the
            # highest value of min-value(result(board, action)).
            results = {}
            for action in actions(board):
                results[action] = min_value(result(board, action))
            
            return max(results, key=results.get)

        #	Minimizer
        elif player(board) == O:
            # picks action in actions(board) that produces the
            # lowest value of max-value(result(board, action)).
            results = {}
            for action in actions(board):
                results[action] = max_value(result(board, action))

            return min(results, key=results.get)
        
        else: 
        # return None if the game is over
            return None

