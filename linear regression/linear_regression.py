import numpy as np

# -----------------------------
# Sample dataset
# -----------------------------
# Columns: [Area, Bedrooms, Bathrooms, Age]
X = np.array([      #np arrays are faster than python lists cuz C styled with math functionality
    [1000, 2, 1, 10],
    [1500, 3, 2, 5],
    [1800, 3, 2, 8],
    [2400, 4, 3, 2],
    [3000, 5, 4, 1],
    [3500, 5, 4, 3],
], dtype=float)

# Target: Price (lakhs)
y = np.array([40, 65, 70, 95, 130, 140], dtype=float)

# -----------------------------
# Feature Scaling (IMPORTANT)
# -----------------------------
# Gradient descent works MUCH better if features are scaled
X_mean = X.mean(axis=0)   #axis = 0 means along the column, 4D is impossible to imagine how do people work in 3D+..
X_std = X.std(axis=0)     #standard deviation = root of variance, variance = 1/n sigma(x - mean)^2

X_scaled = (X - X_mean) / X_std  # x.shape = (6,4) but x_mean.shape = (4,) so numpy uses broadcastiing
# X_mean expanded to (1,4) then (6,4):
# [
#   [mean1, mean2, mean3, mean4],
#   [mean1, mean2, mean3, mean4],
#   [mean1, mean2, mean3, mean4],
#   [mean1, mean2, mean3, mean4],
#   [mean1, mean2, mean3, mean4],
#   [mean1, mean2, mean3, mean4]
# ]



# -----------------------------
# Add bias column for intercept
# -----------------------------
m = len(X_scaled)  # number of samples
X_b = np.c_[np.ones((m, 1)), X_scaled]  # np.c_ concats columnwise, np.ones((m,1)) create m rows 1 column and initialize to 1, X_b shape: (m, n_features+1)
print(X_b, "\n")

# -----------------------------
# Gradient Descent
# -----------------------------
theta = np.zeros(X_b.shape[1])  #X_b.shape[1] = (6, 5)[1] = 5 => [bias, w1, w2, w3, w4]

learning_rate = 0.1
epochs = 600

for epoch in range(epochs):
    hypothesis = X_b @ theta # @ is matrix multiplication, (6,5) * (5,) => (6,5) * (1,5) => (6,)
    error = hypothesis - y   # by default column wise operations

    gradients = (2/m) * (X_b.T @ error) # .T means taking transpose, 2/m cuz we using mean squared error, not half MSE
    theta = theta - learning_rate * gradients

    if epoch % 100 == 0:
        mse = np.mean(error ** 2)
        print(f"Epoch {epoch}, MSE = {mse:.4f}")

print("\nFinal theta values:") #the bias seems to be constant, 90.0
print(theta)

# -----------------------------
# Predict on new house
# -----------------------------
# New house: 2000 sqft, 3 bed, 2 bath, 4 years old
new_house = np.array([[2100, 3, 2, 4]], dtype=float)

# scale using training mean/std
new_house_scaled = (new_house - X_mean) / X_std

# add bias
new_house_b = np.c_[np.ones((1, 1)), new_house_scaled]

predicted_price = new_house_b @ theta
print("\nPredicted price (lakhs):", predicted_price[0])
