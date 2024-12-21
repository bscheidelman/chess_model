import tkinter as tk
from tkinter import Canvas, PhotoImage, messagebox
import chess
import os
import numpy as np
from tensorflow.keras.models import load_model
import random

MODEL_PATH = "chess_probability_model_32k.h5"
model = load_model(MODEL_PATH)
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

def encode_board_and_moves(board):
    fen = board.fen()
    position_encoding = encode_positions([fen])[0]  
    legal_moves = list(board.legal_moves)
    move_encoding = [hash(str(move)) % 1000 for move in legal_moves]
    move_encoding = move_encoding[:64] + [0] * (64 - len(move_encoding))
    return np.array([position_encoding]), np.array([move_encoding]), legal_moves

def predict_move(model, board):
    position_encoding, move_encoding, legal_moves = encode_board_and_moves(board)
    probabilities = model.predict([position_encoding, move_encoding])[0]

    normalized_probabilities = [probabilities[i] for i in range(len(legal_moves))]
    total = sum(normalized_probabilities)
    normalized_probabilities = [p / total for p in normalized_probabilities]

    chosen_move = random.choices(
        population=legal_moves,
        weights=normalized_probabilities,
        k=1
    )[0]

    move_probabilities = {move: probabilities[i] for i, move in enumerate(legal_moves)}
    return chosen_move, move_probabilities

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
        move, probabilities = predict_move(model, board)
        board.push(move)
        draw_board()
        print("AI move probabilities:", {str(m): p for m, p in probabilities.items()})
        highest_prob_move = max(probabilities, key=probabilities.get)
        highest_prob = probabilities[highest_prob_move]

        print(f"Highest probability move: {highest_prob_move} with probability {highest_prob:.4f}")
        print(f"Chosen move: {move} with probability {probabilities[move]:.4f}")

    canvas.bind("<Button-1>", on_drag_start)
    canvas.bind("<ButtonRelease-1>", on_drag_end)

    pieces = load_piece_images()

    draw_board()
    root.mainloop()

if __name__ == "__main__":
    start_game()