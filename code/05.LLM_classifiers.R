library(ellmer)
library(httr2)
library(tidyverse)
source('code/utils.R')

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

Here are three examples of a review not talking about schedule: 
  1. 'i have been working at alta resources for sunrun part-time for more than a year the brea office is a tight family, everyone became friends and free lunch on thursdays.'
  2. 'people are boring and the workplace conversations are limited. some people are not very good at their jobs and they're allowed to stay way too long.'

Reply with exactly one character: 1 if the review mentions schedule, 0 if not.
Do not add any explanation or punctuation."

reviews <- read_csv("data/training_set.csv", show_col_types = FALSE) |> 
  mutate(
    gpt_label = NA_integer_,
    qwen_sm_label = NA_integer_,
    qwen_lg_label = NA_integer_
  )

#====================================
# Closed model: GPT-5.4
#====================================
if (FALSE) {
for (i in seq_len(nrow(reviews))) {
  chat <- chat_openai(system_prompt = system_prompt,
                      model = "gpt-5.4",
                      api_args = list(temperature=0),
                      echo='none')
  reviews$gpt_label[i] <- as.integer(trimws(chat$chat(reviews$doc[i])))
}

cr_gpt <- classification_report(reviews$label, reviews$gpt_label)
}

#====================================
# Open pretrained model: Qwen3.5:9b
#====================================
# NOTE: ellmer's chat_ollama() uses the OpenAI-compatible endpoint which
# ignores think=false for Qwen3 models. We call the native API directly.

ollama_classify <- function(doc, system_prompt, model,
                            base_url = "http://localhost:11434") {
  body <- list(
    model = model,
    messages = list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = doc)
    ),
    think = FALSE,
    stream = FALSE,
    options = list(temperature = 0, num_predict = 3)
  )
  resp <- request(base_url) |>
    req_url_path("/api/chat") |>
    req_body_json(body) |>
    req_timeout(120) |>
    req_perform()
  content <- resp_body_json(resp)$message$content
  as.integer(trimws(content))
}

n <- nrow(reviews)
for (i in seq_len(n)) {
  reviews$qwen_sm_label[i] <- ollama_classify(reviews$doc[i], system_prompt, "qwen3.5")
  cat(sprintf("\r  Qwen3.5:9b — %d/%d (%.0f%%)", i, n, i / n * 100))
}
cat("\n")

cr_qwen_sm <- classification_report(reviews$label, reviews$qwen_sm_label)


#====================================
# Open finetuned model: Qwen3.5:9b 
#====================================



