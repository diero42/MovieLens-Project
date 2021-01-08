knitr::opts_chunk$set(echo = TRUE)

#START OF PROVIDED CODE

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# END OF PROVIDED CODE

# START OF ORIGINAL CODE

# Our test set will be 20% of remaining edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)

# We split train and test sets from edx set along test index
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Ensure userId and movieId in test set are also in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Define function to calculate Root Mean Square Error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))}

# Begin process of regularization by using cross-validation to find the tuning parameter (Lambda) that provides the lowest RMSE
# We will accomplish this by testing a series of possible Lambdas
tuning_params <- seq(0, 10, 0.25)
test_errors <- sapply(tuning_params, function(l){
  
  # First we find the overall average rating for all movies (mu)
  mu <- mean(train_set$rating)
  
  # Next we use our regularization equation to find average movie ratings (b_i)
  regularized_movie_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Again we use our regularization equation to find the user-specific effect (b_u)
  regularized_user_avgs <- train_set %>% 
    left_join(regularized_movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Finally we will use these calculated values to form our predicted ratings
  predicted_ratings <- test_set %>% 
    left_join(regularized_movie_avgs, by='movieId') %>%
    left_join(regularized_user_avgs, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  # We now test each prediction against the test set and return all calculated RMSE values
  return(RMSE(predicted_ratings, test_set$rating))
})

# Checking the calculated RMSEs, we select the tuning parameter which resulted in the lowest RMSE value and assign it to Lambda
lambda <- tuning_params[which.min(test_errors)]

# Now that we have selected the appropriate tuning parameter to regularize our prediction, we can repeat the process on our edx set and compare it against the hold-out validation set
mu <- mean(edx$rating)

regularized_movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda))

regularized_user_avgs <- edx %>% 
  left_join(regularized_movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- validation %>% 
  left_join(regularized_movie_avgs, by='movieId') %>%
  left_join(regularized_user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#We now calculate the final RMSE of our model using our predicted value from the edx set and the final validation set
model_rmse <- RMSE(predicted_ratings, validation$rating)
plot(tuning_params,test_errors,xlab = "Lambda",ylab = "RMSE")
tuning_params[which.min(test_errors)]
RMSE(predicted_ratings, validation$rating)