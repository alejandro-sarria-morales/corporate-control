library(ellmer)
library(tidyverse)

system_prompt <- "You are a research assistant classifying job reviews.
Determine whether the review contains any mention of work schedule.

Mentions of schedule include:
  - Working hours (long hours, short hours, specific shifts)
  - Overtime (mandatory or voluntary)
  - Flexibility or rigidity of hours
  - Availability requirements (on-call, weekends, holidays)
  - Stability or predictability of working hours

Reply with exactly one character: 1 if the review mentions schedule, 0 if not.
Do not add any explanation or punctuation."

# Load data
df <- read_csv("data/labelled_docs.csv", show_col_types = FALSE)

# Classify
df$llm_schedule <- NA_integer_

for (i in seq_len(nrow(df))) {
  chat <- chat_ollama(system_prompt = system_prompt,
                      model = "llama3.1:8b")
  df$llm_schedule[i] <- as.integer(trimws(chat$chat(df$doc[i])))
}

write_csv(df, "data/llm_classified.csv")
