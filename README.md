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
My solution uses a LangChain pipeline to extract receipt values with a vision model and calculate the two folder totals in Python. In build_chain(), ChatPromptTemplate is connected to ChatDeepSeek using deepseek-v4-flash-vision-exp with temperature=0. The chain is created once, and answer_queries(chain, images) processes every supplied receipt sequentially. For each image, the provided image_data_url(path) helper creates a Base64 data URL, which is passed to the multimodal prompt. The model is instructed to return a JSON object containing final_payment_hkd, subtotal_hkd, and discounts_hkd. These represent the payment after ROUNDING, the SUBTOTAL before ROUNDING, and a list of discount amounts. The prompt excludes ROUNDING from discounts and requests null for unreadable fields. Python parses the response, validates the required values, and converts monetary amounts to Decimal with two decimal places. Negative discount entries are converted to positive values, while unreadable or invalid required amounts stop processing and identify the receipt filename. Query 1 sums final_payment_hkd across all receipts. Query 2 sums subtotal_hkd plus every discount amount, without adding ROUNDING. Finally, answer_queries() returns a dictionary keyed by the two exact question strings, with one HKD amount per response. The provided runner writes the two query rows to results.csv and checks them against ground_truth.json when it is available.
<img width="1376" height="1015" alt="image" src="https://github.com/user-attachments/assets/f7bd111f-bafa-4d2c-a497-3da1a046ccec" />



