import tkinter as tk
from tkinter import Canvas, PhotoImage, messagebox
import chess
import os
import numpy as np
from tensorflow.keras.models import load_model
import random

MODEL_PATH = "chess_probability_model_32k.h5"
model = load_model(MODEL_PATH)

# Encoding functions

def encode_positions(fens):
    encoded_positions = []
    for fen in fens:
        parts = fen.split()
        board = parts[0]
        turn = 1 if parts[1] == 'w' else 0
        board_encoded = []
        for char in board:
            if char.isdigit():
                board_encoded.extend([0] * int(char))
            elif char.isalpha():
                board_encoded.append(ord(char) - ord('a') + 1)
        board_encoded.extend([0] * (64 - len(board_encoded)))
        encoded_positions.append(board_encoded + [turn])
    return np.array(encoded_positions)

def encode_legal_moves_and_probs(moves, probs):
    max_moves = 64
    legal_moves_vector = np.zeros(max_moves, dtype=np.float32)
    prob_vector = np.zeros(max_moves, dtype=np.float32)

    for move, prob in zip(moves, probs.values()):
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj.to_square < max_moves:
                move_index = move_obj.to_square
                legal_moves_vector[move_index] = 1
                prob_vector[move_index] = prob
        except Exception as e:
            print(f"Error encoding move {move}: {e}")

    return legal_moves_vector, prob_vector

def encode_previous_moves(previous_moves):
    max_history = 10
    max_moves = 64
    history_vector = np.zeros((max_history, max_moves), dtype=np.float32)

    for i, move in enumerate(previous_moves[-max_history:]):
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj.to_square < max_moves:
                move_index = move_obj.to_square
                history_vector[i, move_index] = 1
        except Exception as e:
            print(f"Error encoding previous move {move}: {e}")

    return history_vector.flatten()

# Prediction function

def predict_move(model, board):
    """
    Predicts the best move for the current board state using the model.
    Handles cases where no previous moves or probabilities are available.
    """
    fen = board.fen()
    position_encoding = encode_positions([fen])[0]

    legal_moves = list(board.legal_moves)
    legal_move_strs = [move.uci() for move in legal_moves]
    if not legal_move_strs:
        print("No legal moves available.")
        return None

    # Assign uniform probabilities if none are available
    probs = {move: 1 / len(legal_moves) for move in legal_move_strs}

    move_encoding, prob_vector = encode_legal_moves_and_probs(legal_move_strs, probs)
    previous_moves_encoding = encode_previous_moves([])  # Provide empty history if not available
    move_number_encoding = np.array([[board.fullmove_number]])

    inputs = [
        np.array([position_encoding]),
        np.array([move_encoding]),
        move_number_encoding,
        np.array([previous_moves_encoding]),
    ]

    probabilities = model.predict(inputs)[0]

    # Filter probabilities to only consider legal moves
    legal_probabilities = {move: probabilities[chess.Move.from_uci(move).to_square] for move in legal_move_strs}
    print(legal_probabilities)

    moves = list(legal_probabilities.keys())
    weights = list(legal_probabilities.values())

    random_move_str = random.choices(moves, weights=weights, k=1)[0]  # `k=1` returns a single move
    random_move = chess.Move.from_uci(random_move_str)
    
    return random_move


# Game setup

def start_game():
    root = tk.Tk()
    root.title("Chess AI")

    board = chess.Board()
    canvas = Canvas(root, width=480, height=480)
    canvas.pack()
    square_size = 60

    def load_piece_images():
        loaded_pieces = {}
        names = ["p", "r", "n", "b", "q", "k"]
        for color, prefix in [(chess.WHITE, "l"), (chess.BLACK, "d")]:
            for piece in names:
                filename = os.path.join("images", f"Chess_{piece}{prefix}t60.png")
                try:
                    loaded_pieces[piece + ("w" if color else "b")] = PhotoImage(file=filename)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return loaded_pieces

    def draw_board():
        canvas.delete("all")
        for rank in range(8):
            for file in range(8):
                x0 = file * square_size
                y0 = rank * square_size
                x1 = x0 + square_size
                y1 = y0 + square_size
                fill = "white" if (rank + file) % 2 == 0 else "gray"
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill)

        for square, piece in board.piece_map().items():
            x = chess.square_file(square) * square_size
            y = (7 - chess.square_rank(square)) * square_size
            img = pieces.get(piece.symbol().lower() + ("w" if piece.color else "b"), None)
            if img:
                canvas.create_image(x + square_size // 2, y + square_size // 2, image=img)

    def on_drag_start(event):
        x, y = event.x, event.y
        file = x // square_size
        rank = 7 - (y // square_size)
        square = chess.square(file, rank)
        if board.piece_at(square):
            canvas.data = square

    def on_drag_end(event):
        x, y = event.x, event.y
        file = x // square_size
        rank = 7 - (y // square_size)
        target_square = chess.square(file, rank)

        try:
            move = chess.Move(canvas.data, target_square)
            if move in board.legal_moves:
                board.push(move)
                draw_board()
                ai_move()
            else:
                messagebox.showerror("Invalid Move", "That move is not valid.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def ai_move():
        try:
            move = predict_move(model, board)
            if move is not None:
                board.push(move)
                draw_board()
            else:
                print("No valid move found. AI skips its turn.")
        except Exception as e:
            print(f"Error during AI move: {e}")

    canvas.bind("<Button-1>", on_drag_start)
    canvas.bind("<ButtonRelease-1>", on_drag_end)

    pieces = load_piece_images()
    draw_board()
    root.mainloop()

if __name__ == "__main__":
    start_game()
