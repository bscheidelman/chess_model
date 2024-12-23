import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import chess

def parse_chess_data(file_path):
    positions, legal_moves, probs, move_numbers, previous_moves = [], [], [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            entry = eval(line.strip())
            positions.append(entry['fen'])
            legal_moves.append(entry['legal_moves'])
            probs.append(dict(zip(entry['probs keys'], entry['probs values'])))
            move_numbers.append(entry.get('move_number', 0))  
            previous_moves.append(entry.get('previous_moves', []))  
    return positions, legal_moves, probs, move_numbers, previous_moves


def preprocess_chess_data(positions, legal_moves, probs, move_numbers, previous_moves):
    position_encodings = encode_positions(positions)
    legal_moves_encodings, prob_distributions = encode_legal_moves_and_probs(legal_moves, probs)
    move_number_encodings = np.array(move_numbers).reshape(-1, 1)
    previous_move_encodings = encode_previous_moves(previous_moves)
    return position_encodings, legal_moves_encodings, np.array(prob_distributions), move_number_encodings, previous_move_encodings

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
                piece_value = ord(char.lower()) - ord('a') + 1
                board_encoded.append(piece_value if char.islower() else -piece_value)
        board_encoded.extend([0] * (64 - len(board_encoded)))
        encoded_positions.append(board_encoded + [turn])
    return np.array(encoded_positions)

def encode_legal_moves_and_probs(moves_list, probs_list):
    max_moves = 64
    legal_moves_encoded = []
    prob_distributions = []

    for moves, probs in zip(moves_list, probs_list):
        legal_moves_vector = np.zeros(max_moves, dtype=np.float32)
        prob_vector = np.zeros(max_moves, dtype=np.float32)

        for move, prob in zip(moves, probs.values()):
            try:
                move_obj = chess.Move.from_uci(move)
                move_index = move_obj.to_square
                legal_moves_vector[move_index] = 1
                prob_vector[move_index] = prob
            except Exception as e:
                print(f"Error encoding move {move}: {e}")

        if prob_vector.sum() > 0:
            prob_vector /= prob_vector.sum()

        legal_moves_encoded.append(legal_moves_vector)
        prob_distributions.append(prob_vector)

    return np.array(legal_moves_encoded), np.array(prob_distributions)

def encode_previous_moves(previous_moves_list):
    max_history = 10
    max_moves = 64
    encoded_history = []
    for moves in previous_moves_list:
        history_vector = np.zeros((max_history, max_moves), dtype=np.float32)
        for i, move in enumerate(moves[-max_history:]):
            try:
                move_obj = chess.Move.from_uci(move)
                move_index = move_obj.to_square
                history_vector[i, move_index] = 1
            except Exception as e:
                print(f"Error encoding previous move {move}: {e}")
        encoded_history.append(history_vector.flatten())
    return np.array(encoded_history)

def build_model(input_dim_position, input_dim_moves, input_dim_move_number, input_dim_previous_moves, output_dim):
    position_input = Input(shape=(65,), name='Position_Input')
    position_dense = Dense(256, activation='relu', name='Position_Dense')(position_input)
    position_dense = BatchNormalization()(position_dense)

    moves_input = Input(shape=(64,), name='Moves_Input')
    moves_dense = Dense(256, activation='relu', name='Moves_Dense')(moves_input)
    moves_dense = BatchNormalization()(moves_dense)

    move_number_input = Input(shape=(1,), name='Move_Number_Input')
    move_number_dense = Dense(64, activation='relu', name='Move_Number_Dense')(move_number_input)

    previous_moves_input = Input(shape=(640,), name='Previous_Moves_Input')  # 10 moves x 64 max moves
    previous_moves_dense = Dense(128, activation='relu', name='Previous_Moves_Dense')(previous_moves_input)

    concatenated = Concatenate()([position_dense, moves_dense, move_number_dense, previous_moves_dense])
    dense_1 = Dense(512, activation='relu')(concatenated)
    dense_1 = Dropout(0.4)(dense_1)
    dense_2 = Dense(256, activation='relu')(dense_1)
    dense_2 = Dropout(0.4)(dense_2)
    output = Dense(output_dim, activation='softmax', name='Output')(dense_2)

    model = Model(inputs=[position_input, moves_input, move_number_input, previous_moves_input], outputs=output)
    return model

def train_and_evaluate_model(model, position_data, move_data, prob_data, move_numbers, previous_moves, epochs=100, batch_size=128):
    x_train_pos, x_test_pos, x_train_moves, x_test_moves, x_train_probs, x_test_probs, x_train_nums, x_test_nums, x_train_prev, x_test_prev = train_test_split(
        position_data, move_data, prob_data, move_numbers, previous_moves, test_size=0.2, random_state=42
    )

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.KLDivergence()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        [x_train_pos, x_train_moves, x_train_nums, x_train_prev], x_train_probs,
        validation_data=([x_test_pos, x_test_moves, x_test_nums, x_test_prev], x_test_probs),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping]
    )

    results = model.evaluate([x_test_pos, x_test_moves, x_test_nums, x_test_prev], x_test_probs)
    print("Test Loss, Accuracy, KL Divergence:", results)

if __name__ == "__main__":
    file_path = "chess_data.txt"
    positions, legal_moves, probs, move_numbers, previous_moves = parse_chess_data(file_path)
    position_data, move_data, prob_data, move_number_data, previous_move_data = preprocess_chess_data(positions, legal_moves, probs, move_numbers, previous_moves)

    position_input_dim = 65  # Board and turn
    move_input_dim = 64  # Fixed size for legal moves
    move_number_input_dim = 1  # Single scalar for move number
    previous_moves_input_dim = 640  # 10 moves x 64 max moves
    output_dim = 64  # Fixed size for legal moves

    model = build_model(position_input_dim, move_input_dim, move_number_input_dim, previous_moves_input_dim, output_dim)
    train_and_evaluate_model(model, position_data, move_data, prob_data, move_number_data, previous_move_data)

    model.save("chess_probability_model_32k.h5")
