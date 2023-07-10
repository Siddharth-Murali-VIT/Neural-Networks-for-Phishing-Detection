import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
dataset = pd.read_csv('testdata.csv')

# Split into input (X) and output (y) variables
X = dataset.iloc[:, 0]  # Input
y = dataset.iloc[:, 1]  # Output

# Perform label encoding on the input column
label_encoder_X = LabelEncoder()
X_encoded = label_encoder_X.fit_transform(X)

# Perform label encoding on the output column
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(1,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the keras model on the training dataset
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

# Make predictions on the testing dataset
predictions = (model.predict(X_test) > 0.5).astype(int)

# Perform label decoding on the testing set for printing
decoded_X_test = label_encoder_X.inverse_transform(X_test)
decoded_y_test = label_encoder_y.inverse_transform(y_test)
decoded_predictions = label_encoder_y.inverse_transform(predictions)

# Measure accuracy on the testing set
accuracy = (predictions == y_test).mean()

# Summarize the predictions on the testing set
for i in range(len(X_test)):
    print('Input: %s => Predicted Output: %d (Expected Output: %s)' % (decoded_X_test[i], decoded_predictions[i], decoded_y_test[i]))

# Print the accuracy
print('Accuracy on the testing set:', accuracy)