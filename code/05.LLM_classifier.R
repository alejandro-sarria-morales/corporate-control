library(ellmer)
library(tidyverse)
library(progress)

reviews <- read_csv("data/training_set.csv", show_col_types = FALSE)

system_prompt <- "You are a research assistant classifying job reviews.
Determine whether the review contains any mention of work schedule.

Mentions of schedule include:
  - Working hours (long hours, short hours, specific shifts)
  - Overtime (mandatory or voluntary)
  - Flexibility or rigidity of hours
  - Availability requirements (on-call, weekends, holidays)
  - Stability or predictability of working hours
  
Here are two examples of a review talking about schedule: 
  1. 'long hours and a lot of work'
  2. 'they offer flexible hours and the staff is very nice.'

Here are two examples of a review not talking about schedule: 
  1. 'i have been working at alta resources for sunrun part-time for more than a year the brea office is a tight family, everyone became friends and free lunch on thursdays.'
  2. 'people are boring and the workplace conversations are limited. some people are not very good at their jobs and they're allowed to stay way too long.'

Reply with exactly one character: 1 if the review mentions schedule, 0 if not.
Do not add any explanation or punctuation."

reviews$llm_schedule <- NA_integer_

pb <- progress_bar$new(
  format = " Labelling reviews with Llama [:bar] :percent",
  total=length(reviews$llm_schedule),
  clear = FALSE)

for (i in seq_len(nrow(reviews))) {
  pb$tick()
  chat <- chat_ollama(system_prompt = system_prompt,
                      model = "llama3.1:8b",
                      echo='none')
  reviews$llm_schedule[i] <- as.integer(trimws(chat$chat(reviews$doc[i])))
}

# Compute metrics against hand labels
pred <- reviews$llm_schedule
true <- reviews$label

tp <- sum(pred == 1 & true == 1)
fp <- sum(pred == 1 & true == 0)
fn <- sum(pred == 0 & true == 1)
tn <- sum(pred == 0 & true == 0)

accuracy  <- (tp + tn) / length(true)
recall    <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- Classiffier Evaluation ---\n")
cat(sprintf("Accuracy:  %.2f\n", accuracy))
cat(sprintf("Recall:    %.2f\n", recall))
cat(sprintf("Precision: %.2f\n", precision))
cat(sprintf("F1:        %.2f\n", f1))
