# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
The solution divides the work between the vision model and Python. `build_chain()` creates the `ChatDeepSeek` model with the required `deepseek-v4-flash-vision-exp` model name, sets the temperature to zero for more consistent extraction, and connects it to a `ChatPromptTemplate`. The prompt asks for only three fields in JSON: the final amount paid after `ROUNDING`, the `SUBTOTAL` before rounding, and a list of every discount, promotion, or coupon. It explicitly tells the model to leave rounding out of the discount list and to return `null` when a value is unreadable rather than guess.

`answer_queries(chain, images)` processes the supplied receipt images one at a time. For each image, the existing `image_data_url()` helper reads the file and encodes it as a Base64 data URL. The chain sends that image and the extraction instructions together to the vision model. Python then removes optional Markdown code fences, parses the JSON, checks that the required fields contain valid amounts, and converts the amounts to `Decimal` so that currency totals are added accurately. If a response is malformed or a required amount cannot be read, the program reports the receipt filename instead of silently using a guessed value.

After extraction, Python calculates the two answers separately. Question 1 adds the receipts' `final_payment_hkd` values, which already include each receipt's rounding adjustment. Question 2 adds each `subtotal_hkd` to the absolute value of every discount in `discounts_hkd`; it deliberately excludes `ROUNDING`. The function returns one formatted HKD amount for each exact question string. The provided runner writes those answers to `results.csv` and compares them with `ground_truth.json` when that file is present.
<img width="1376" height="1015" alt="image" src="https://github.com/user-attachments/assets/f7bd111f-bafa-4d2c-a497-3da1a046ccec" />



