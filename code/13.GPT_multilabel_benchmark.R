# Closed-model benchmark for the two-dimension (schedule / job control) classifier.
#
# Runs gpt-5.4 over the SAME frozen held-out test rows (set == 0) the fine-tuned model
# is evaluated on, using the SAME system prompt file, then writes predictions to
# data/trainingfinal/gpt_preds.csv. Score them with the identical metric code via:
#
#   python code/12.evaluate_multilabel.py --preds_csv data/trainingfinal/gpt_preds.csv --name gpt-5.4
#
# CPU-only job (just OpenAI API calls) -- no GPU or DCC allocation needed. Run it the
# way you normally run the GPT benchmarks (locally / on CPU); needs OPENAI_API_KEY.
# DCC is reserved for fine-tuning and evaluating the open models.

library(ellmer)
library(tidyverse)

MODEL       <- "gpt-5.4"
DATA_CSV    <- "data/trainingfinal/labelled.csv"
PROMPT_FILE <- "code/prompts/multilabel_system_prompt.txt"
OUT_CSV     <- "data/trainingfinal/gpt_preds.csv"

SYSTEM_PROMPT <- str_trim(paste(readLines(PROMPT_FILE, warn = FALSE), collapse = "\n"))

# Parse a model reply into c(schedule, job_control), permissive about stray characters.
parse_two_digit <- function(x) {
  digits <- str_split(str_remove_all(as.character(x), "[^0-9]"), "")[[1]]
  digits <- suppressWarnings(as.integer(digits))
  digits <- digits[!is.na(digits)]
  s <- if (length(digits) >= 1) as.integer(digits[1] != 0) else 0L
  j <- if (length(digits) >= 2) as.integer(digits[2] != 0) else 0L
  c(s, j)
}

reviews <- read_csv(DATA_CSV, show_col_types = FALSE) |>
  filter(set == 0) |>
  mutate(sched_pred = NA_integer_, ctrl_pred = NA_integer_, raw = NA_character_)

cat(sprintf("Benchmarking %s on %d held-out test reviews...\n", MODEL, nrow(reviews)))

n <- nrow(reviews)
for (i in seq_len(n)) {
  # Fresh chat each row so no conversation history leaks between reviews.
  chat <- chat_openai(system_prompt = SYSTEM_PROMPT, model = MODEL,
                      api_args = list(temperature = 0), echo = "none")
  raw <- tryCatch(trimws(chat$chat(reviews$review_text[i])),
                  error = function(e) { message("  row ", i, ": ", conditionMessage(e)); NA_character_ })
  reviews$raw[i] <- raw
  pj <- parse_two_digit(raw)
  reviews$sched_pred[i] <- pj[1]
  reviews$ctrl_pred[i]  <- pj[2]
  if (i %% 25 == 0) cat(sprintf("  %d/%d done\n", i, n))
}

write_csv(reviews |> select(id, sched_pred, ctrl_pred, raw), OUT_CSV)
cat(sprintf("\nSaved GPT predictions to %s\n", OUT_CSV))
cat("Score them with:\n")
cat(sprintf("  python code/12.evaluate_multilabel.py --preds_csv %s --name %s\n", OUT_CSV, MODEL))
