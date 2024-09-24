
# Import Libraries
library(dplyr)
library(tidyverse)
library(caret)
library(VIM)
library(ggplot2)
library(leaflet)
library(randomForest)
library(ranger)
library(rpart)  
library(e1071)
library(Metrics) 


# Import the dataset
file <- ("C:/Users/Shruti/OneDrive/INTERNSHIP STUFF/Projects/R casestudy/Dataset .csv")

data <- read.csv(file)

# LEVEL1 TASK1

# See the content
glimpse(data)


# Explore the dataset
str(data) # structure of the dataset

summary(data) # summary statistics

dim(data) # number of rows and columns

# Check for missing values
missing_values <- colSums(is.na(data))
print(missing_values)

# Columns with missing values
missing_columns <- names(missing_values[missing_values > 0])

# Fill missing values with mean
for (col in missing_columns) {
  mean_value <- mean(data[[col]], na.rm = TRUE)
  data[[col]][is.na(data[[col]])] <- mean_value
}

# Verify that missing values are filled
missing_values_after_fill <- colSums(is.na(data))
print(missing_values_after_fill)


# Convert numeric columns
data$Price.range <- as.numeric(data$Price.range)
data$Aggregate.rating <- as.numeric(data$Aggregate.rating)
data$Votes <- as.numeric(data$Votes)

# Distribution of Aggregate rating
summary(data$Aggregate.rating)

# Plot histogram using ggplot2
ggplot(data, aes(x = Aggregate.rating)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Aggregate Rating", x = "Aggregate Rating", y = "Frequency")

# LEVEL1 TASK2

# Calculate basic statistical measures for numerical columns
numerical_columns <- c("Average.Cost.for.two", "Price.range", "Aggregate.rating", "Votes")
summary_data <- sapply(data[, numerical_columns], function(x) c(Mean = mean(x), Median = median(x), SD = sd(x)))
summary_data

# Explore distribution of categorical variables
# Country Code
table(data$Country.Code)

# City
table(data$City)

# Cuisines
table(data$Cuisines)

# Identify top cuisines
top_cuisines <- head(sort(table(data$Cuisines), decreasing = TRUE), 10)
top_cuisines

# Identify top cities with the highest number of restaurants
top_cities <- head(sort(table(data$City), decreasing = TRUE), 10)
top_cities

# LEVEL1 TASK3

# Create a leaflet map with restaurant locations
leaflet(data) %>%
  addTiles() %>%
  addMarkers(~Longitude, ~Latitude, popup = ~Restaurant.Name)

# Distribution of restaurants across cities
city_distribution <- table(data$City)
barplot(city_distribution, main = "Restaurants Across Cities", xlab = "City", ylab = "Number of Restaurants")

# Distribution of restaurants across countries
country_distribution <- table(data$Country.Code)
barplot(country_distribution, main = "Restaurants Across Countries", xlab = "Country Code", ylab = "Number of Restaurants")


# Correlation between latitude, longitude, and rating
correlation_matrix <- cor(data[, c("Longitude", "Latitude", "Aggregate.rating")])
correlation_matrix


# LEVEL2 TASK1

# Calculate the percentage of restaurants offering table booking
table_booking_percentage <- mean(data$Has.Table.booking == "Yes") * 100
cat("Percentage of restaurants offering table booking:", round(table_booking_percentage, 2), "%\n")

# Calculate the percentage of restaurants offering online delivery
online_delivery_percentage <- mean(data$Has.Online.delivery == "Yes") * 100
cat("Percentage of restaurants offering online delivery:", round(online_delivery_percentage, 2), "%\n")

# Calculate the average rating of restaurants with table booking
average_rating_with_booking <- mean(data[data$Has.Table.booking == "Yes", "Aggregate.rating"], na.rm = TRUE)

# Calculate the average rating of restaurants without table booking
average_rating_without_booking <- mean(data[data$Has.Table.booking == "No", "Aggregate.rating"], na.rm = TRUE)

# Print the comparison
cat("Average rating of restaurants with table booking:", round(average_rating_with_booking, 2), "\n")
cat("Average rating of restaurants without table booking:", round(average_rating_without_booking, 2), "\n")

# Calculate the percentage of restaurants offering online delivery for each price range
online_delivery_by_price_range <- aggregate(Has.Online.delivery ~ Price.range, 
                                            data = data, 
                                            FUN = function(x) mean(x == "Yes") * 100)

# Display the results
online_delivery_by_price_range <- cbind(online_delivery_by_price_range[,1], online_delivery_by_price_range[,2])
colnames(online_delivery_by_price_range) <- c("Price Range", "Online Delivery Percentage")
print(online_delivery_by_price_range)


# LEVEL2 TASK2   

# Determine the most common price range among all the restaurants
most_common_price_range <- names(sort(table(data$Price.range), decreasing = TRUE))[1]

print("Most Common Price Range Among All Restaurants:")
print(most_common_price_range)

# Calculate the average rating for each price range
average_rating_by_price_range <- data %>%
  group_by(Price.range) %>%
  summarise(average_rating = mean(Aggregate.rating, na.rm = TRUE))

print("Average Rating for Each Price Range:")
print(average_rating_by_price_range)

# Identify the color that represents the highest average rating among different price ranges
highest_average_rating_color <- average_rating_by_price_range$Price.range[which.max(average_rating_by_price_range$average_rating)]

print("Color Representing the Highest Average Rating:")
print(highest_average_rating_color)
# LEVEL2 TASK3

# Extract additional features
data$RestaurantNameLength <- nchar(data$Restaurant.Name)
data$AddressLength <- nchar(data$Address)

# Create new features by encoding categorical variables
data$HasTableBooking <- as.integer(data$Has.Table.booking == "Yes")
data$HasOnlineDelivery <- as.integer(data$Has.Online.delivery == "Yes")

# Print head of the modified dataset to verify changes
head(data)


# LEVEL3 TASK1

# Step 1: Prepare the data
# Select relevant features and preprocess the data as needed

features <- c("Average.Cost.for.two", "Price.range", "Votes", "Has.Table.booking", "Has.Online.delivery")
target <- "Aggregate.rating"

# Remove rows with missing values in selected features

data <- na.omit(data[, c(features, target)])

# Step 2: Split the dataset into training and testing sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$Aggregate.rating, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Step 3: Build regression models
# Experiment with different algorithms (e.g., linear regression, decision trees, random forest)
lm_model <- train(Aggregate.rating ~ ., data = train_data, method = "lm")
dt_model <- train(Aggregate.rating ~ ., data = train_data, method = "rpart")
rf_model <- train(Aggregate.rating ~ ., data = train_data, method = "rf")

# Step 4: Train the models
# Models are trained during the train() function call in Step 3.

# Step 5: Evaluate the models

lm_pred <- predict(lm_model, newdata = test_data)
dt_pred <- predict(dt_model, newdata = test_data)
rf_pred <- predict(rf_model, newdata = test_data)

# Step 6: Compare performance

lm_rmse <- RMSE(lm_pred, test_data$Aggregate.rating)
dt_rmse <- RMSE(dt_pred, test_data$Aggregate.rating)
rf_rmse <- RMSE(rf_pred, test_data$Aggregate.rating)

print("Root Mean Squared Error (RMSE) for Different Models:")
print(paste("Linear Regression:", lm_rmse))
print(paste("Decision Tree:", dt_rmse))
print(paste("Random Forest:", rf_rmse))


# LEVEL3 TASK2

# Cuisine-Rating Analysis
cuisine_rating <- data %>%
  group_by(Cuisines) %>%
  summarise(mean_rating = mean(Aggregate.rating), 
            total_votes = sum(Votes))

# Visualize the relationship between cuisine and rating
# Select top N cuisines based on mean rating
top_n_cuisines <- head(cuisine_rating[order(cuisine_rating$mean_rating, decreasing = TRUE), ], 10)

# Plot horizontal bar plot for top N cuisines
ggplot(top_n_cuisines, aes(x = mean_rating, y = reorder(Cuisines, mean_rating))) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Top 10 Cuisines by Mean Rating",
       x = "Mean Rating",
       y = "Cuisine")

# Popular Cuisine Analysis
most_popular_cuisines <- cuisine_rating %>%
  arrange(desc(total_votes)) %>%
  head(10)  # Top 10 most popular cuisines based on votes

# Print the top 10 most popular cuisines
print(most_popular_cuisines)

# High-Rating Cuisine Analysis
high_rating_cuisines <- cuisine_rating %>%
  arrange(desc(mean_rating)) %>%
  head(10)  # Top 10 cuisines with highest average ratings

# Print the top 10 high-rating cuisines
print(high_rating_cuisines)


# LEVEL3 TASK3

# Distribution of Ratings
ggplot(data, aes(x = Aggregate.rating)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Ratings", x = "Rating", y = "Frequency")

# Comparison of Average Ratings by Cuisine
cuisine_avg_ratings <- data %>%
  group_by(Cuisines) %>%
  summarise(mean_rating = mean(Aggregate.rating))

# Select top N cuisines based on average rating
top_n_cuisines <- cuisine_avg_ratings %>%
  arrange(desc(mean_rating)) %>%
  head(10)

# Plot horizontal bar plot for top N cuisines
ggplot(top_n_cuisines, aes(x = mean_rating, y = reorder(Cuisines, mean_rating))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Top 10 Cuisines by Average Rating",
       x = "Mean Rating",
       y = "Cuisine") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Comparison of Average Ratings by City
city_avg_ratings <- data %>%
  group_by(City) %>%
  summarise(mean_rating = mean(Aggregate.rating))

ggplot(city_avg_ratings, aes(x = reorder(City, mean_rating), y = mean_rating)) +
  geom_bar(stat = "identity", fill = "grey") +
  labs(title = "Average Ratings by City", x = "City", y = "Mean Rating") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Relationship between Features and Target Variable
# For example, let's visualize the relationship between average cost for two and ratings
ggplot(data, aes(x = `Average.Cost.for.two`, y = Aggregate.rating)) +
  geom_point(color = "red") +
  labs(title = "Relationship between Average Cost for Two and Ratings", x = "Average Cost for Two", y = "Rating")
