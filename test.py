import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

# Load weather data into a DataFrame
# Assuming the weather data is loaded from a CSV file named 'weather_data.csv'
weather_data = pd.read_csv('tavg_train.csv')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(weather_data[['tavg']])

# Define function to create input sequences and corresponding targets
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

# Define hyperparameters
sequence_length = 10
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Prepare the sequences and targets
x, y = create_sequences(scaled_data, sequence_length)

# Convert data to PyTorch tensors
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    inputs = Variable(x).to(device)
    targets = Variable(y).to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions for the next month
num_days_to_predict = 30
predicted_temperatures = []

# Set initial input sequence for prediction
input_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

# Make predictions for each day in the next month
for _ in range(num_days_to_predict):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_sequence).to(device)
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy()

    # Append the predicted temperature to the list
    predicted_temperatures.append(prediction[0][0])

    # Update the input sequence for the next prediction
    input_sequence = np.append(input_sequence[:, 1:, :], [[prediction]], axis=1)

# Inverse transform the predicted temperatures to get the original scale
predicted_temperatures = scaler.inverse_transform(np.array(predicted_temperatures).reshape(-1, 1))

# Print the predicted temperatures for the next month
print('Predicted temperatures for the next month:')
for i, temp in enumerate(predicted_temperatures):
    print(f"Day {i+1}: {temp[0]}")
