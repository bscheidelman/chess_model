import requests
import chess
import json
import time
from collections import deque

LICHESS_API_URL = "https://explorer.lichess.ovh/lichess"

fen_cache = {}

def fetch_lichess_data(fen):
    if fen in fen_cache:
        print("Fen Already Stored")
        return fen_cache[fen]

    params = {
        'fen': fen,
        'topGames': 0,
        'ratings': [2000],  
    }
    
    response = requests.get(LICHESS_API_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        moves = data.get("moves", [])
        total_games = sum(move['white'] + move['black'] + move['draws'] for move in moves)
        move_probs = {
            move['uci']: (move['white'] + move['black'] + move['draws']) / total_games
            for move in moves
        }
        fen_cache[fen] = (move_probs, total_games)
        return move_probs, total_games
    else:
        #print(f"Error: Received status code {response.status_code}")
        fen_cache[fen] = (None, 0)
        return None, 0


def traverse_and_collect_data(fen, required_samples):
    board = chess.Board(fen)
    move_probs, total_games = fetch_lichess_data(fen)

    if not move_probs or total_games < 100:
        return

    queue = deque([(board, move_probs, 0, [])])  

    i = 0
    start_time = time.time()

    while queue and (len(queue) + i) <= required_samples:
        current_board, current_move_probs, move_number, previous_moves = queue.popleft()
        
        print(f"Processing board: {current_board.fen()}")
        print(f"Data count: {i+len(queue)}/{required_samples}")

        with open("chess_data.txt", "a") as file:
            file.write(json.dumps({
                "fen": current_board.fen(),
                "legal_moves": [str(move) for move in current_board.legal_moves],
                "probs keys": [str(move) for move in current_move_probs.keys()],
                "probs values": [float(prob) for prob in current_move_probs.values()],
                "move_number": move_number, 
                "previous_moves": previous_moves  
            }) + "\n")

        i += 1
        total_processed = i + len(queue)
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / total_processed) * required_samples if total_processed > 0 else 0
        time_remaining = max(0, estimated_total_time - elapsed_time)

        hours, remainder = divmod(time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Elapsed time: {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s, Estimated time to completion: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        if i >= required_samples:
            print("Required samples collected, stopping.")
            return

        if (len(queue) + i) <= required_samples + 1:
            for move in sorted(current_move_probs, key=current_move_probs.get, reverse=True):
                current_board.push(chess.Move.from_uci(move))
                new_move_probs, new_total_games = fetch_lichess_data(current_board.fen())

                if new_total_games >= 100:
                    new_previous_moves = previous_moves[-9:] + [move] 
                    queue.append((current_board.copy(), new_move_probs, move_number + 1, new_previous_moves))

                current_board.pop()

    print("Traversal complete.")

traverse_and_collect_data(chess.STARTING_FEN, 32000)
