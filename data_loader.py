import requests
import chess
import json
from collections import deque


LICHESS_API_URL = "https://explorer.lichess.ovh/lichess"

fen_cache = {}

def fetch_lichess_data(fen):
    if fen in fen_cache:
        return fen_cache[fen]

    params = {
        'fen': fen,
        'topGames': 0,
    }
    response = requests.get(LICHESS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        moves = data.get("moves", [])
        total_games = sum(move['white'] + move['black'] + move['draws'] for move in moves)
        move_probs = {
            chess.Move.from_uci(move['uci']): (move['white'] + move['black'] + move['draws']) / total_games
            for move in moves
        }
        fen_cache[fen] = (move_probs, total_games)
        return move_probs, total_games
    else:
        fen_cache[fen] = (None, 0)
        return None, 0



from collections import deque


def traverse_and_collect_data(fen, required_samples):
    board = chess.Board(fen)
    move_probs, total_games = fetch_lichess_data(fen)
    
    if not move_probs or total_games < 100: return
    
    queue = deque([(board, move_probs)])

    i = 0
    while queue and i <= required_samples:
        i+= 1
        current_board, current_move_probs = queue.popleft()
        
        print(f"Processing board: {current_board.fen()}")
        print(f"Data count: {i}/{required_samples}")
        
        with open("chess_data.txt", "a") as file:
            file.write(json.dumps({"fen": current_board.fen(), "legal_moves": [str(move) for move in current_board.legal_moves], "probs keys": [str(move) for move in current_move_probs.keys()], "probs values": [float(prob) for prob in current_move_probs.values()]}) + "\n")
            
        if i >= required_samples:
            print("Required samples collected, stopping.")
            return
        
        print("Current Queue Length:", len(queue))
        if (len(queue) + i) <= required_samples + 5:
            for move in sorted(current_move_probs, key=current_move_probs.get, reverse=True):
                current_board.push(move)
                move_probs_check, total_games_check = fetch_lichess_data(current_board.fen())
                
                if total_games_check >= 100:
                    print(f"Adding new state: {current_board.fen()}")
                    queue.append((current_board.copy(), move_probs_check))
                    
                current_board.pop() 

    print("Traversal complete.")



traverse_and_collect_data(chess.STARTING_FEN, 32000)
