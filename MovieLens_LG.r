# title: "MovieLens Recommendation System"
# author: "Loren Grooms"
# date: "05/14/2021"

knitr::opts_chunk$set(echo = TRUE)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# Download and store zipped file.
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Remove delimiters and label columns appropriately.
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Remove delimiters using a utility for processing strings, label
# columns appropriately.
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
tags <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/tags.dat")), "\\::", 4)
colnames(tags) <- c("userId", "movieId", "tag", "timestamp")

# Convert to data frames after converting strings to appropriate data types.
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
tags <- as.data.frame(tags) %>% mutate(userId = as.numeric(userId),
                                       movieId = as.numeric(movieId),
                                       tag = as.character(tag),
                                       timestamp = as.numeric(timestamp)) %>% 
                                       select(tag,timestamp)

# Join tables into primary dataset.
movielens <- left_join(ratings, tags, by = "timestamp")
movielens <- left_join(movielens, movies, by = "movieId")

movielens[with(movielens,order(tag))] %>% head()

cat("Rows: ",nrow(movielens))
cat("Columns: ",ncol(movielens))

movielens %>% group_by(rating) %>% summarize(count = n()) %>% ggplot(aes(rating,count)) + geom_line() + labs(title = "Fig1", x = "Star Rating", y = "Count")

movielens %>% count(movieId) %>% ggplot(aes(n)) + geom_histogram(bins = 30, color = "black") + labs(title = "Fig2", x = "Ratings per Movie", y = "Count")

movielens %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(bins = 30, color = "black") + labs(title = "Fig3", x = "Ratings per User", y = "Count")

tags <- table(movielens$tag) %>% as.data.frame() %>% arrange(desc(Freq))
tags %>% arrange(desc(Freq)) %>% top_n(10) %>% ggplot(aes(x = reorder(Var1, -Freq),y = Freq)) + geom_bar(stat="identity",color = "black") + labs(title = "Fig4", x = "Unique Tags", y = "Count") + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

# Validation set will be 10% of MovieLens data, split by rating.
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Keep only rows which have corresponding rows in edx set.
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows that didn't have a corresponding edx row back into edx set.
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Test set will be 10% of remaining edx set, split by rating.
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Keep only rows which have corresponding rows in training set.
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Remove unnecessary variables.
rm(dl, ratings, movies, tags, test_index, temp, movielens, removed)

# Define function to calculate Root Mean Square Error.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))}

# Begin process of regularization by using cross-validation to find the tuning
# parameter (Lambda) that provides the lowest RMSE.
# We will accomplish this by testing a series of possible Lambdas.
tuning_params <- seq(0, 10, 0.25)
test_errors <- sapply(tuning_params, function(l){
  
  # First we find the overall average rating for all movies (mu).
  mu <- mean(train_set$rating)
  
  # Next we use our regularization equation to find average movie
  # ratings (b_i).
  regularized_movie_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Again we use our regularization equation to find the user-specific
  # effect (b_u).
  regularized_user_avgs <- train_set %>% 
    left_join(regularized_movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Finally we will use these calculated values to form our predicted ratings.
  predicted_ratings <- test_set %>% 
    left_join(regularized_movie_avgs, by='movieId') %>%
    left_join(regularized_user_avgs, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  # We now test each prediction against the test set and return all calculated
  # RMSE values.
  return(RMSE(predicted_ratings, test_set$rating))
})

# Checking the calculated RMSEs, we select the tuning parameter which resulted
# in the lowest RMSE value and assign it to Lambda.
lambda <- tuning_params[which.min(test_errors)]

plot(tuning_params,test_errors,xlab = "Lambda",ylab = "RMSE")

tuning_params[which.min(test_errors)]

# Now that we have selected the appropriate tuning parameter to regularize
# our prediction, we can repeat the process on our edx set and compare it
# against the hold-out validation set.
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

# We now calculate the final RMSE of our model using our predicted value
# from the edx set and the final validation set.
model_rmse <- RMSE(predicted_ratings, validation$rating)

cat(model_rmse)