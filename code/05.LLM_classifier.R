library(ellmer)
library(tidyverse)
library(progress)

system_prompt <- "You are a research assistant classifying job reviews.
Determine whether the review contains any mention of work schedule.

Mentions of schedule include:
  - Working hours (long hours, short hours, specific shifts)
  - Overtime (mandatory or voluntary)
  - Flexibility or rigidity of hours
  - Availability requirements (on-call, weekends, holidays)
  - Stability or predictability of working hours
  
Here is an example of a review talking about schedule: 'always be caution in working, high risk work'

Here is an example of a review not talking about schedule: 'i have been working at alta resources for sunrun part-time for more than a year the brea office is a tight family, everyone became friends and free lunch on thursdays.'

Reply with exactly one character: 1 if the review mentions schedule, 0 if not.
Do not add any explanation or punctuation."


#===============================
# Evaluation on subset B
#===============================
subset_b <- read_csv("data/schedule_subset_b.csv", show_col_types = FALSE)

subset_b$llm_schedule <- NA_integer_

pb <- progress_bar$new(
  format = " Labelling reviews with Llama [:bar] :percent",
  total=length(subset_b$llm_schedule),
  clear = FALSE)

for (i in seq_len(nrow(subset_b))) {
  pb$tick()
  chat <- chat_ollama(system_prompt = system_prompt,
                      model = "llama3.1:8b",
                      echo='none')
  subset_b$llm_schedule[i] <- as.integer(trimws(chat$chat(subset_b$doc[i])))
}

# Compute metrics against hand labels
pred <- subset_b$llm_schedule
true <- subset_b$hand_label

tp <- sum(pred == 1 & true == 1)
fp <- sum(pred == 1 & true == 0)
fn <- sum(pred == 0 & true == 1)
tn <- sum(pred == 0 & true == 0)

accuracy  <- (tp + tn) / length(true)
recall    <- tp / (tp + fn)
precision <- tp / (tp + fp)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- Subset B Evaluation ---\n")
cat(sprintf("Accuracy:  %.3f\n", accuracy))
cat(sprintf("Recall:    %.3f\n", recall))
cat(sprintf("Precision: %.3f\n", precision))
cat(sprintf("F1:        %.3f\n", f1))
