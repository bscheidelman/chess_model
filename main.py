import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Parse chess data
def parse_chess_data(file_path):
    positions, legal_moves, probs = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            entry = eval(line.strip())
            positions.append(entry['fen'])
            legal_moves.append(entry['legal_moves'])
            probs.append(dict(zip(entry['probs keys'], entry['probs values'])))
    return positions, legal_moves, probs

# Preprocess chess data
def preprocess_chess_data(positions, legal_moves, probs):
    position_encodings = encode_positions(positions)
    legal_moves_encodings, prob_distributions = encode_legal_moves_and_probs(legal_moves, probs)
    return position_encodings, legal_moves_encodings, np.array(prob_distributions)

# Improved encoder functions
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

def encode_legal_moves_and_probs(moves_list, probs_list):
    max_moves = 64
    legal_moves_encoded = []
    prob_distributions = []

    for moves, probs in zip(moves_list, probs_list):
        move_vector = [hash(move) % 1000 for move in moves]  # Replace with move encoding
        legal_moves_encoded.append(move_vector + [0] * (max_moves - len(move_vector)))

        prob_vector = [probs.get(move, 0) for move in moves]
        prob_vector += [0] * (max_moves - len(prob_vector))
        prob_distributions.append(prob_vector)

    return np.array(legal_moves_encoded), np.array(prob_distributions)

# Define the neural network model
def build_model(input_dim_position, input_dim_moves, output_dim):
    position_input = Input(shape=(65,), name='Position_Input')
    position_dense = Dense(256, activation='relu', name='Position_Dense')(position_input)
    position_dense = BatchNormalization()(position_dense)

    moves_input = Input(shape=(64,), name='Moves_Input')
    moves_dense = Dense(256, activation='relu', name='Moves_Dense')(moves_input)
    moves_dense = BatchNormalization()(moves_dense)

    concatenated = Concatenate()([position_dense, moves_dense])
    dense_1 = Dense(512, activation='relu')(concatenated)
    dense_1 = Dropout(0.4)(dense_1)
    dense_2 = Dense(256, activation='relu')(dense_1)
    dense_2 = Dropout(0.4)(dense_2)
    output = Dense(output_dim, activation='softmax', name='Output')(dense_2)

    model = Model(inputs=[position_input, moves_input], outputs=output)
    return model

# Train-test split and evaluation setup
def train_and_evaluate_model(model, position_data, move_data, prob_data, epochs=100, batch_size=128):
    x_train_pos, x_test_pos, x_train_moves, x_test_moves, y_train, y_test = train_test_split(
        position_data, move_data, prob_data, test_size=0.2, random_state=42
    )

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.KLDivergence()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        [x_train_pos, x_train_moves], y_train,
        validation_data=([x_test_pos, x_test_moves], y_test),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping]
    )

    results = model.evaluate([x_test_pos, x_test_moves], y_test)
    print("Test Loss, Accuracy, KL Divergence:", results)

# Main script
if __name__ == "__main__":
    file_path = "chess_data.txt"
    positions, legal_moves, probs = parse_chess_data(file_path)
    position_data, move_data, prob_data = preprocess_chess_data(positions, legal_moves, probs)

    # Determine input dimensions
    position_input_dim = 65  # Board and turn
    move_input_dim = 64  # Fixed size for legal moves
    output_dim = 64  # Fixed size for legal moves

    # Build and train model
    model = build_model(position_input_dim, move_input_dim, output_dim)
    train_and_evaluate_model(model, position_data, move_data, prob_data)

    # Save model
    model.save("chess_probability_model_32k.h5")
