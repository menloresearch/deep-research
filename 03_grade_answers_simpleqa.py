import concurrent.futures
import csv
import os
import re
import sys  # Import sys for exiting

from dotenv import load_dotenv
from openai import APIError, AuthenticationError, OpenAI, RateLimitError  # Import specific exceptions

load_dotenv(override=True)

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
Use code with caution.
Python
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
Use code with caution.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
Use code with caution.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
Use code with caution.
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

# Define valid grade letters and corresponding descriptions
CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))

# Define specific error states for internal use
ERROR_NO_API_KEY = "ERROR_NO_API_KEY"
ERROR_API_CALL_FAILED = "ERROR_API_CALL_FAILED"
ERROR_UNEXPECTED_LLM_RESPONSE = "ERROR_UNEXPECTED_LLM_RESPONSE"


def clean_predicted_answer(text: str) -> str:
    """Removes common prefixes from the predicted answer."""
    if text.startswith("**Final answer:**\n"):
        return text.replace("**Final answer:**\n", "", 1).strip()
    if text.startswith("**Final answer:**"):
        return text.replace("**Final answer:**", "", 1).strip()
    return text.strip()


def call_llm_grader(question: str, target: str, predicted_answer: str) -> str:
    """
    Calls the OpenRouter API (via OpenAI library) with the formatted GRADER_TEMPLATE
    and returns the LLM's raw response string or an error state string.
    """
    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(f"ERROR: OPENROUTER_API_KEY environment variable not set. Cannot call API for grading.", file=sys.stderr)
        # Return a specific error state string
        return ERROR_NO_API_KEY

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        completion = client.chat.completions.create(
            # extra_headers={
            #     "HTTP-Referer": "<YOUR_SITE_URL>", # Replace with your site URL
            #     "X-Title": "<YOUR_SITE_NAME>",     # Replace with your site name
            # },
            model="openai/gpt-4o",  # Or your preferred model
            messages=[{"role": "user", "content": grader_prompt}],
            max_tokens=10,  # We only expect A, B, or C, plus a little buffer
            temperature=0.0,  # For deterministic grading
        )
        llm_response_content = completion.choices[0].message.content
        if llm_response_content:
            return llm_response_content.strip()
        else:
            # LLM returned empty content - treat as an unexpected response
            return ERROR_UNEXPECTED_LLM_RESPONSE
    except (AuthenticationError, APIError, RateLimitError) as e:
        # Specific OpenAI/OpenRouter errors
        print(f"API Error calling LLM for question '{question[:50]}...': {e}", file=sys.stderr)
        return ERROR_API_CALL_FAILED
    except Exception as e:
        # Catch any other unexpected errors during the API call
        print(f"Unexpected Error calling LLM for question '{question[:50]}...': {e}", file=sys.stderr)
        return ERROR_API_CALL_FAILED


def process_row(row_dict: dict[str, str]) -> dict[str, str]:
    """
    Processes a single row from the CSV:
    - Extracts question, gold_answer, and predicted_answer.
    - Cleans the predicted_answer.
    - Calls the grader.
    - Appends grade information to the row.
    """
    question = row_dict.get("query", "")
    gold_answer = row_dict.get("gold_answer", "")
    predicted_answer_raw = row_dict.get("answer", "")

    cleaned_predicted_answer = clean_predicted_answer(predicted_answer_raw)

    # --- Determine Grade ---
    grade_letter: str
    grade_description: str

    if not question or not gold_answer:
        # Cannot grade if essential info is missing in the input data
        grade_letter = "C"  # Map missing input to NOT_ATTEMPTED
        grade_description = "Input Missing Essential Data (Q/A)"
        # print(f"Warning: Missing question or gold_answer for a row. Grading as NOT_ATTEMPTED.", file=sys.stderr) # Avoid flooding stdout/stderr in parallel
    elif not cleaned_predicted_answer:
        # If predicted answer is empty after cleaning
        grade_letter = "C"  # Map empty prediction to NOT_ATTEMPTED
        grade_description = "Predicted Answer Empty"
        # print(f"Warning: Empty predicted answer for question: '{question[:50]}...'. Grading as NOT_ATTEMPTED.", file=sys.stderr) # Avoid flooding stdout/stderr in parallel
    else:
        # Call the LLM grader
        llm_raw_response = call_llm_grader(question, gold_answer, cleaned_predicted_answer)

        # --- Interpret LLM Response / Error State ---
        if llm_raw_response == ERROR_NO_API_KEY:
            grade_letter = "ERROR"
            grade_description = "No API Key Configured"
            # Note: A separate check in main will exit early if this happens
        elif llm_raw_response == ERROR_API_CALL_FAILED:
            grade_letter = "ERROR"
            grade_description = "LLM API Call Failed"
        elif llm_raw_response == ERROR_UNEXPECTED_LLM_RESPONSE:
            grade_letter = "ERROR"
            grade_description = "LLM Returned Unexpected/Empty Response"
            print(
                f"Warning: LLM returned unexpected/empty content for question '{question[:50]}...'. Original LLM response: '{llm_raw_response}'. Marking as ERROR.",
                file=sys.stderr,
            )
        else:
            # Attempt to match expected A, B, or C
            match = re.search(r"\b(A|B|C)\b", llm_raw_response)
            if match:
                grade_letter = match.group(0)
                grade_description = CHOICE_LETTER_TO_STRING.get(grade_letter, "UNKNOWN_GRADE")  # Should not be UNKNOWN
            else:
                # LLM response was not one of the specific errors, but also not A, B, or C
                grade_letter = "ERROR"
                grade_description = "LLM Did Not Return A/B/C"
                print(
                    f"Warning: LLM returned unexpected response format for question '{question[:50]}...'. Original LLM response: '{llm_raw_response}'. Marking as ERROR.",
                    file=sys.stderr,
                )

    # --- Prepare Output Row ---
    output_row = row_dict.copy()
    output_row["grade_letter"] = grade_letter
    output_row["grade_description"] = grade_description
    return output_row


def main():
    # Hardcoded input path based on your provided output
    input_csv_path = "simpleqa_432_simpleqa_14b_deepresearch_v0.2_200s.csv"
    output_csv_path = "graded_simpleqa_432_simpleqa_14b_deepresearch_v0.2_200s.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}", file=sys.stderr)
        sys.exit(1)  # Exit if input file is missing

    rows_to_process: list[dict[str, str]] = []
    fieldnames_from_reader: list[str] = []
    try:
        with open(input_csv_path, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames:
                fieldnames_from_reader = list(reader.fieldnames)
            else:
                fieldnames_from_reader = []  # Handles case of empty file or no headers
            for row in reader:
                rows_to_process.append(row)
    except Exception as e:
        print(f"Error reading input CSV {input_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)  # Exit on file read error

    total_rows = len(rows_to_process)
    if not total_rows:
        if not fieldnames_from_reader:  # Truly empty file (no headers, no rows)
            print(f"Input CSV file {input_csv_path} is empty or has no headers. Nothing to process.")
        else:  # File has headers but no data rows
            print(f"No data rows found in {input_csv_path} (only headers). Nothing to process.")
        return  # Exit if no data to process

    print(f"Starting grading for {total_rows} rows...")

    results: list[dict[str, str]] = []  # Use a list, append results in order
    num_workers = 8
    processed_count = 0  # Counter for processed rows
    error_occurred_critical = False  # Flag for critical errors like API key missing

    # Using list comprehension to get futures and maintain original order
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Store futures along with their original index
        futures = [(executor.submit(process_row, row), i) for i, row in enumerate(rows_to_process)]

        # Collect results in order
        for future, original_index in futures:
            try:
                processed_row_data = future.result()
                results.append(processed_row_data)  # Append results as they complete, but process in order later
                # Check for critical errors returned by process_row
                if processed_row_data.get("grade_description") == "No API Key Configured":
                    print("\nCritical Error: API Key is not configured. Stopping processing.", file=sys.stderr)
                    error_occurred_critical = True
                    # In ProcessPoolExecutor with as_completed, it's hard to stop gracefully
                    # We'll just mark the flag and handle it after the loop
            except Exception as exc:
                print(f"\nError processing row index {original_index}: {exc}", file=sys.stderr)
                error_row = rows_to_process[original_index].copy()
                error_row["grade_letter"] = "ERROR"
                error_row["grade_description"] = f"Processing Error: {exc}"
                results.append(error_row)  # Add error result
            finally:
                processed_count += 1
                # Print progress, using \r to overwrite the line for a cleaner output
                # Check the flag before printing progress to avoid clobbering critical error message
                if not error_occurred_critical:
                    print(f"Processed {processed_count} / {total_rows} rows...", end="\r")

    # Print a newline after the progress indicator loop finishes (if not already done by critical error)
    if not error_occurred_critical:
        print("\nAll rows submitted to workers.")

    # Wait for all results to be collected (already done by iterating through futures)
    # Sort results by original index if necessary (appending might keep a mixed order depending on completion time)
    # Let's sort the results list based on the original index if we appended them in a mixed order
    # A better way might be to collect results into a list of size total_rows and place them by index
    # Let's revise the result collection slightly.

    # Re-doing the result collection using a list of fixed size to ensure order
    results: list[dict[str, str] | None] = [None] * total_rows
    futures = {}  # Reset futures dictionary for ordered collection
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        for i, row in enumerate(rows_to_process):
            futures[executor.submit(process_row, row)] = i

        processed_count = 0
        error_occurred_critical = False

        # Collect results as they complete, placing them in the results list by index
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            try:
                processed_row_data = future.result()
                results[original_index] = processed_row_data

                # Check for critical errors returned by process_row
                if processed_row_data and processed_row_data.get("grade_description") == "No API Key Configured":
                    print(
                        "\nCritical Error: API Key is not configured. Stopping submission of *new* tasks.",
                        file=sys.stderr,
                    )
                    error_occurred_critical = True
                    # Cancel remaining pending futures (doesn't guarantee immediate stop)
                    for f in futures:
                        if not f.done():
                            f.cancel()

            except concurrent.futures.CancelledError:
                # Handle cancelled futures gracefully
                pass  # Just ignore cancelled tasks; they are already marked as None in results

            except Exception as exc:
                print(f"\nError processing row index {original_index}: {exc}", file=sys.stderr)
                error_row = rows_to_process[original_index].copy()
                error_row["grade_letter"] = "ERROR"
                error_row["grade_description"] = f"Processing Error: {exc}"
                results[original_index] = error_row

            finally:
                processed_count += 1
                # Print progress, using \r to overwrite the line for a cleaner output
                # Check the flag before printing progress
                if not error_occurred_critical:
                    print(f"Processed {processed_count} / {total_rows} rows...", end="\r")
                elif processed_count % 10 == 0:  # Print progress less frequently after critical error
                    print(f"Processed {processed_count} / {total_rows} rows (processing remaining tasks)...", end="\r")

    # Print a newline after the progress indicator loop finishes
    print("\nAll processing attempted.")

    # Filter out None results if cancellation left gaps (shouldn't happen if all submitted tasks complete or are cancelled)
    final_results: list[dict[str, str]] = [res for res in results if res is not None]

    if len(final_results) != total_rows:
        print(
            f"Warning: Expected {total_rows} results, but got {len(final_results)}. Some tasks may have failed or been cancelled unexpectedly.",
            file=sys.stderr,
        )

    # Print results to console (optional, but helps see what happened)
    print("\n--- Graded Results Summary ---")
    grade_counts = {}
    error_counts = {}
    for row in final_results:
        grade = row.get("grade_description", "UNKNOWN")
        if grade == "CORRECT" or grade == "INCORRECT" or grade == "NOT_ATTEMPTED":
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        else:
            # This is an error state
            error_counts[grade] = error_counts.get(grade, 0) + 1

    if grade_counts:
        print("Standard Grades:")
        for grade, count in grade_counts.items():
            print(f"  {grade}: {count}")

    if error_counts:
        print("\nError Summary:")
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")

    if not final_results:
        print("No results were processed.")
    print("--- End of Graded Results Summary ---\n")

    # --- Write to Output CSV ---
    # Determine output fieldnames: Start with input headers, add grade columns
    output_fieldnames: list[str]
    if fieldnames_from_reader:
        output_fieldnames = list(fieldnames_from_reader)
    elif final_results:
        # Fallback if no headers from reader (unlikely if input file wasn't empty/malformed)
        output_fieldnames = list(final_results[0].keys())
    else:
        # No headers and no results (should have been caught earlier)
        print("Cannot determine output CSV headers: No source headers and no processed data.", file=sys.stderr)
        return  # Exit if cannot determine headers

    # Ensure grade columns are in the output headers
    if "grade_letter" not in output_fieldnames:
        output_fieldnames.append("grade_letter")
    if "grade_description" not in output_fieldnames:
        output_fieldnames.append("grade_description")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(final_results)  # Write the collected results

        print(f"Successfully processed {len(final_results)} rows. Output saved to {output_csv_path}")

    except Exception as e:
        print(f"Error writing output CSV {output_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)  # Exit on output file write error

    # Exit with a non-zero status if critical errors occurred
    if error_occurred_critical:
        print("\nScript finished with critical errors.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
