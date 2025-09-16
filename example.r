# Load dataset
data(mtcars)

# View first few rows
head(mtcars)

# Summary statistics
summary(mtcars)

# Correlation between variables
cor(mtcars$mpg, mtcars$wt)

# Scatter plot (miles per gallon vs weight)
plot(mtcars$wt, mtcars$mpg,
     main = "MPG vs Car Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles Per Gallon",
     pch = 19,
     col = "blue")

# Add a regression line
model <- lm(mpg ~ wt, data = mtcars)
abline(model, col = "red", lwd = 2)

# Print model summary
summary(model)

# Predict MPG for new car weights
new_weights <- data.frame(wt = c(2.5, 3.0, 3.5))
predictions <- predict(model, new_weights)
print(predictions)