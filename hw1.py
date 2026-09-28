#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Add it to your local .env file "
            "or set it as an environment variable."
        )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=500,
        max_retries=2,
        timeout=60,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You read supermarket receipts carefully. Return only one valid "
                "JSON object with these fields: final_payment_hkd, "
                "subtotal_hkd, and discounts_hkd. Use numbers or numeric strings "
                "in HKD, without currency symbols or thousands separators. "
                "final_payment_hkd is the actual amount paid after ROUNDING. "
                "subtotal_hkd is the receipt's SUBTOTAL before ROUNDING. "
                "discounts_hkd is a list containing every discount, promotion, "
                "or coupon amount as a positive number. Do not include ROUNDING "
                "in discounts_hkd. If a field cannot be read, return null for "
                "that field instead of guessing. Do not include explanations, "
                "item prices, percentages, or any other numbers.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read the payment, subtotal, and all discount lines from this receipt.",
                    },
                    {"type": "image_url", "image_url": "{image_url}"},
                ],
            ),
        ]
    )
    return prompt | model
    return None


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
 if not images:
        raise ValueError("At least one receipt image is required.")

    def parse_receipt_response(response: Any, image: Path) -> dict[str, Any]:
        text = response_text(response)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"The model did not return a JSON object for {image.name}.")
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as error:
                raise ValueError(f"The model returned invalid JSON for {image.name}.") from error

        if not isinstance(data, dict):
            raise ValueError(f"The model response for {image.name} must be a JSON object.")
        return data

    def money_amount(value: Any, field: str, image: Path) -> Decimal:
        if value is None or isinstance(value, bool):
            raise ValueError(f"The model could not read {field} from {image.name}.")
        try:
            amount = Decimal(str(value).replace(",", "").replace("HK$", "").strip())
        except (InvalidOperation, ValueError):
            raise ValueError(f"The model returned an invalid {field} for {image.name}.") from None
        if not amount.is_finite():
            raise ValueError(f"The model returned an invalid {field} for {image.name}.")
        if amount < 0 and field == "discount":
            amount = amount.copy_abs()
        elif amount < 0:
            raise ValueError(f"The model returned an invalid {field} for {image.name}.")
        return amount.quantize(Decimal("0.01"))

    total_paid = Decimal("0.00")
    total_without_discounts = Decimal("0.00")

    for image in images:
        response = chain.invoke({"image_url": image_data_url(image)})
        receipt = parse_receipt_response(response, image)

        paid = money_amount(receipt.get("final_payment_hkd"), "final payment", image)
        subtotal = money_amount(receipt.get("subtotal_hkd"), "subtotal", image)
        discounts = receipt.get("discounts_hkd")
        if not isinstance(discounts, list):
            raise ValueError(f"The model did not return a discount list for {image.name}.")

        discount_total = sum(
            (money_amount(value, "discount", image) for value in discounts),
            Decimal("0.00"),
        )
        total_paid += paid
        total_without_discounts += subtotal + discount_total
    return {QUERY_1: DUMMY_RESPONSE, QUERY_2: DUMMY_RESPONSE}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
