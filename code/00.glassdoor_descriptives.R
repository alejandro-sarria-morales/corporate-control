#------------
# set up    
#------------
library(tidyverse)
library(lubridate)
library(gt)
library(stringi)

d <- read_csv('data/glassdoor_reviews.csv') |>
  select(-...1)

#------------------
# descriptive plots   
#------------------

##------------------
## Reviews histogram
##------------------

g <- d |>
  filter(!is.na(date)) |>
  mutate(y = factor(year(date))) |>
  ggplot(aes(x = y)) +
  geom_bar(fill = "steelblue") +
  labs(x = "", y = "",
       title = "Number of reviews per year") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45,
                                   hjust = 1,
                                   size = 9))

ggsave("figures/review_hist.png", g)

##------------------
## Missingness
##------------------

## Key variables

vars <- c('rating_overall', 'rating_worklife_balance','job_title', 'job_location')

miss_d <- d |> select(date, all_of(vars)) |>
  filter(!is.na(date)) |> 
  mutate(year_ = year(date),
         y_int3 = cut(year_,
                      breaks = seq(2008, 2026, by = 3),
                      right = FALSE,
                      include.lowest = TRUE),
         across(starts_with("rating_"), 
                ~ is.na(.) | (!is.na(.) & . == 0),
                .names = "miss_{.col}"),
         across(setdiff(vars, grep("^rating_", vars, value = TRUE)),
                ~ is.na(.),
                .names = "miss_{.col}")) |> 
  pivot_longer(cols = starts_with("miss_"),
               names_to = "var",
               values_to = "missing") |> 
  mutate(var = sub('^miss_', "", var)) |> 
  group_by(y_int3, var) |> 
  summarise(total = n(),
            n_missing = sum(missing),
            prop_missing = mean(missing))

### Table
miss_d |> 
  gt() |> 
  gtsave("outputs/ratings_missingness.pdf")

### Barplots
g <- miss_d |> 
  ggplot(aes(x=y_int3, y=prop_missing)) + 
  geom_col(fill = "steelblue") +
  facet_wrap(~var) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
  labs(x = "", y = "", title = "Proportion of missing datapoints in key variables",
       subtitle = "Missingness in overall rating is ~0% when date is present")

ggsave("figures/ratings_missingness.png", g)

### Missing dates
d |>
  select(date, all_of(vars)) |> 
  filter(is.na(date)) |> 
  gt() |> 
  gtsave("outputs/missing_dates.pdf")

## Reviews

miss_r <- d |> 
  select("date", "review_pros", "review_cons", "review_advice") |> 
  filter(!is.na(date)) |> 
  mutate(year_ = year(date),
         y_int3 = cut(year_,
                      breaks = seq(2008, 2026, by = 3),
                      right = FALSE,
                      include.lowest = TRUE),
         across(starts_with("review_"), ~ is.na(.), .names = "miss_{.col}")) |> 
  pivot_longer(cols = starts_with("miss_"),
               names_to = "var",
               values_to = "missing") |> 
  mutate(var = sub('^miss_', "", var)) |> 
  group_by(y_int3, var) |> 
  summarise(total = n(),
            n_missing = sum(missing),
            prop_missing = mean(missing))
### Table
miss_r |> 
  gt() |> 
  gtsave("outputs/reviews_missingness.pdf")

### Barplots
g <- miss_r |> 
  ggplot(aes(x=y_int3, y=prop_missing)) + 
  geom_col(fill = "steelblue") +
  facet_wrap(~var, scale="free_y") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
  labs(x = "", y = "", title = "Proportion of missing datapoints in text reviews")

ggsave("figures/reviews_missingness.png", g)



#----------------------
# Text reviews
#----------------------

## Length

g <- d |> 
  select("review_pros", "review_cons", "review_advice") |> 
  filter(if_all(everything(), ~ !is.na(.))) |> 
  pivot_longer(starts_with("review_"),
               names_to = "review_type",
               values_to = "text") |> 
  mutate(num_words = str_count(text, "\\w+")) |>
  ggplot(aes(x=num_words)) + 
  geom_histogram(binwidth=10, fill="steelblue") + 
  facet_grid(rows=vars(review_type))

ggsave("figures/text_hist_full.png", g)

### no outliers
g <- d |> 
  select("review_pros", "review_cons", "review_advice") |> 
  filter(if_all(everything(), ~ !is.na(.))) |> 
  pivot_longer(starts_with("review_"),
               names_to = "review_type",
               values_to = "text") |> 
  mutate(num_words = str_count(text, "\\w+")) |> 
  filter(num_words > 0 & num_words < (mean(num_words) + 2*sd(num_words))) |>
  ggplot(aes(x=num_words)) + 
  geom_histogram(binwidth=1, fill="steelblue") + 
  facet_grid(rows=vars(review_type))

ggsave("figures/text_hist_full_no_outliers.png", g)

## Length descriptives

len_d <- d |>
  select(date, review_pros, review_cons, review_advice) |>
  filter(!is.na(date)) |>
  mutate(year_ = year(date),
         y_int3 = cut(year_,
                      breaks = seq(2008, 2026, by = 3),
                      right = FALSE,
                      include.lowest = TRUE)) |>
  pivot_longer(starts_with("review_"),
               names_to = "review",
               values_to = "text") |>
  filter(!is.na(text)) |>
  mutate(length = str_count(text, "\\w+")) |>
  filter(length > 0) |>
  group_by(y_int3, review) |>
  summarise(avg_len = mean(length),
            sd = sd(length),
            min = min(length),
            max = max(length),
            n = n(),
            .groups = "drop")

### Table
len_d |> 
  gt() |> 
  gtsave("outputs/length_descriptives.pdf")

### Bar plot
g <- len_d |> 
  ggplot(aes(x=y_int3, y=avg_len)) + 
  geom_col(fill = "steelblue") +
  facet_wrap(~review, scale="free_y") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8)) +
  labs(x = "", y = "",
       title = "Average review length in 3 year periods",
       subtitle = "No empty reviews")

ggsave("figures/text_lengths.png", g)


# Final clean data and save
d_clean <- d |>  
  filter(!is.na(date)) |> 
  mutate(across(starts_with("review_"), #fixes issues with encoding
                ~ stri_enc_toutf8(., is_unknown_8bit = TRUE))) |>
  mutate(across(starts_with("review_"), #removes problematic characters (\n, \t, etc.)
                ~ str_replace_all(., "[[:cntrl:]]", " "))) |>
  mutate(across(starts_with("review_"), ~ na_if(., ""))) |> 
  filter(!(is.na(review_pros) &
            is.na(review_cons) &
             is.na(review_advice))) |> 
  mutate(across(starts_with("review_"), #normalizes white space
                ~ str_squish(.))) |>
  mutate(across(starts_with("review_"), #everything to lowercase for consistency
                tolower)) |>
  mutate(across(starts_with("review_"), #add length descriptor for reviews
                ~ str_count(., "\\w+"),
                .names = "len_{.col}"))

write_csv(d_clean, "data/glassdoor_reviews_clean.csv")
  












  
