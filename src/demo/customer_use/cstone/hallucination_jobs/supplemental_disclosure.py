import csv
from openai import OpenAI
from typing import List
from judgeval import JudgmentClient
from judgeval.scorers import GroundednessScorer
from judgeval.data import Example

def extract_data(file_path, 
                               docket_id_col='docket_id', 
                               excerpts_col='excerpts', 
                               llm_response_col='LLM_raw_response', 
                               label_col='correct', 
                               note_col='RZ_note',
                               filter_type='incorrect_only'):
    """
    Extracts data from a CSV file and filters rows based on correctness.
    
    Args:
        file_path (str): Path to the CSV file
        docket_id_col (str): Column name for docket ID
        excerpts_col (str): Column name for excerpts
        llm_response_col (str): Column name for LLM response
        label_col (str): Column name for correctness indicator
        note_col (str): Column name for notes
        filter_type (str): Type of filtering to apply. Options:
                          'incorrect_only': Only return incorrect judgments
                          'correct_only': Only return correct judgments
                          'both': Return both correct and incorrect judgments
    
    Returns:
        list: List of dictionaries containing the extracted data

        {
        "docket_id": "1234567890",
        "excerpts": "excerpts",
        "LLM_raw_response": "LLM_raw_response",
        "correct": "correct",
        "RZ_note": "RZ_note"
        }
    """
    extracted_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            print(f"CSV Headers: {reader.fieldnames}")
            
            for row in reader:
                # Extract the required columns
                data = {
                    'docket_id': row.get(docket_id_col, ''),
                    'excerpts': row.get(excerpts_col, ''),
                    'LLM_raw_response': row.get(llm_response_col, ''),
                    'correct': row.get(label_col, ''),
                    'RZ_note': row.get(note_col, '')
                }
                
                # Filter based on the filter_type
                if filter_type == 'incorrect_only' and data['correct'].upper() == 'FALSE':
                    extracted_data.append(data)
                elif filter_type == 'correct_only' and data['correct'].upper() == 'TRUE':
                    extracted_data.append(data)
                elif filter_type == 'both':
                    extracted_data.append(data)
        
        return extracted_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []


def find_failure_reason(task_instruction: str, llm_response: str, excerpts: str, client=None) -> str:
    """
    Reasons over why the LLM made the wrong judgment.
    
    Args:
        task_instruction: The instruction given for the task
        llm_response: The LLM's response that was incorrect
        excerpts: The context provided to the LLM
        client: Optional OpenAI client instance. If None, a new client will be created.
    
    Returns:
        Analysis of why the LLM made the wrong judgment
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
        
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an analytical lawyer breaking down why an LLM has made a mistake for a task. The task is: {task_instruction}. \n **YOUR JOB** is to analyze the LLM's response to this task and the excerpts it had access to in order to solve the task. The LLM definitely got the task wrong, whether it be hallucinating information, misinterpreting the excepts, or not following the actual task. YOU NEED TO FIND WHY."},
            {"role": "user", "content": f"LLM Response (INCORRECT): {llm_response}\n\nExcerpts the LLM had to work with: {excerpts}"}
        ],
    )
    return response.choices[0].message.content


def find_failure_reasons_parallel(task_instruction: str, data_list: list, response_key: str = 'LLM_raw_response', 
                                 excerpts_key: str = 'excerpts', max_workers: int = 10) -> list:
    """
    Parallelizes the process of finding failure reasons for multiple examples.
    
    Args:
        task_instruction: The instruction given for the task
        data_list: List of dictionaries containing LLM responses and excerpts
        response_key: Key in the dictionaries for LLM responses
        excerpts_key: Key in the dictionaries for excerpts
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of failure reason analyses
    """
    import concurrent.futures
    from openai import OpenAI
    
    client = OpenAI()
    results = []
    
    def process_item(item):
        return find_failure_reason(
            task_instruction=task_instruction,
            llm_response=item.get(response_key, ''),
            excerpts=item.get(excerpts_key, ''),
            client=client
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_list}
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    # Write all the results to a file, with newlines in between
    with open("failure_analysis_results.txt", "w") as f:
        for i, result in enumerate(results):
            f.write(f"Failure Reason for Example {i+1}:\n")
            f.write(result)
            f.write("\n\n" + "-"*80 + "\n\n")
    
    return results
    

def inference(excerpts: str, llm_response: str, task_instruction: str, model: str = "gpt-4o", client=None) -> str:
    """
    {
        "docket_id": "1234567890",
        "excerpts": "excerpts",
        "LLM_raw_response": "LLM_raw_response",
        "correct": "correct",
        "RZ_note": "RZ_note"
        }
    """

    prompt = f"""You are an analytical lawyer breaking down whether or not an LLM has made a hallucination in its response to a task. 
    I'M PRETTY SURE THERE'S A HALLUCINATION, BUT I WANT YOU TO DOUBLE CHECK.
    You will be provided with a set of instructions for the task, a set of excerpts the LLM used to complete the task, and the LLM's response to the task. 
    You must analyze whether based on the information provided, the LLM has made a hallucination. A hallucination is defined as one or more of the following mistakes:
    1. The LLM's response contradicts the information provided in the excerpts
    2. The LLM's response misinterprets the information provided in the excerpts
    3. The LLM's response diverges from the task instructions, e.g. answering a different question than the one provided in the instructions, addressing a different task than the one provided, or accomplishing the main goal but missing on specific, crucial details in the instructions.

    Here are a few things to keep in mind:
    - If an LLM attempts to interpret information from the excerpts via quotes or paraphrasing, YOU MUST PAY CLOSE ATTENTION TO WHETHER THE LLM'S INTERPRETATION IS CORRECT. 
    Do not blindly accept the LLM's interpretation of the evidence as correct -- double check for yourself.
    - If the LLM's response is a quote from the excerpts, you must check whether the quote exists in the excerpts, or if it was fabricated/edited by the LLM.
    - For task instruction following: be very detail-oriented. If the task instructions require the LLM to do multiple things, such as searching for specific evidence and then later use that evidence to answer a question, make sure the LLM has done both things. 
    Additionally, if the task instructions require the LLM to use evidence, check that the LLM uses correctly interpreted/sourced evidence.
    - For task instruction following, pay attention to small conditions in the instructions that the LLM may have missed. 
    For instance, if the instructions dictate "highlight cases of workplace violence in response to a threat", make sure the LLM is highlighting cases of violence in response to a threat, not just cases of violence.

    Again, I'm pretty sure there's a hallucination in the model response, but I COULD BE WRONG!
    ==== THINKING INSTRUCTIONS ====
    First, analyze any possible contradictions and misinterpretations of the information provided in the excerpts.
    Next, analyze whether the LLM's response diverges from the task instructions.
    Finally, based on the above analysis, determine whether the LLM has made a hallucination.

    ==== FORMATTING INSTRUCTIONS ====
    **End your response with <answer>True</answer> if the LLM has made a hallucination, and <answer>False</answer> if it has not made a hallucination.**
    """

    MESSAGES = [
            {"role": "system", "content": prompt},

            {"role": "user", "content": """==== YOUR TURN ====
     TASK INSTRUCTIONS: #### General Task Definition

The provided content contains the following:

- **Target Case** (enclosed in "<target_case>" tags): Basic information about a litigation case, including its caption, filing date, and court.
- **Filing Excerpts** (enclosed in "<filing_excerpts>" tags): Excerpts from SEC filings filed by the defendant company.

You will review the Filing Excerpts and answer two questions: 

1. Did the defendant company (i.e. "the Company", which is the filer of this filing) filed supplemental disclosure or amendment in response to lawsuits?

This is a "Yes" or "No" question. Answer "Yes" if there is explicit mention that the Company has made or planned to make supplemental disclosure or amendment. Otherwise "No".

2. Besides supplemental disclosure, did the defendant company (i.e. "the Company", which is the filer of this filing) reach an explicit agreement with the Target Case parties?

This is a "Yes" or "No" question; for "Yes" you will specify the type of the agreement. Your answer will eventually be one of the followings categories:
- "Yes (Settlement Agreement)": settlement was indicated by an explicit mention of a settlement agreement.
- "Yes (MOU)": settlement was suggested by the parties entering into a "memorandum of understanding" (or "MOU").
- "Yes (Other Agreement)": other than a Settlement Agreement, an MOU, or an agreement to make supplemental disclosure, there is explicit discussion that the Company "entered into an agreement" with the parties of the Target Case, where "agreement" refers to a formal, named agreement type, and the detailed resolution of the agreement is provided.
- "Yes (Implied Agreement)": an agreement is strongly implied but not explicitly stated.
- "No": there is no indication of a settlement or agreement. This includes (but not limited to):
   - there is no discussion of the Target Case.
   - there is no discussion of the outcome of the Target Case.
   - the case was dismissed by the plaintiff, but the excerpts provided no explicit discussion on the reason.
   - there was discussion of an agreement, but the agreement was directly associated with a different case.

#### Thinking Instructions

Follow these steps, stopping as soon as an answer becomes clear. Include all reasoning in your output.

1. **Confirm Case Discussion**: confirm that there are discussion around the Target Case. If the filing did not mention the Target Case at all, you will answer "No" for "supplemental_disclosure" and "reached_agreement" following the Output Instructions. Stop proceeding.

2. **Search for Supplemental Disclosure**: look for explicit mention of the Company agreeing or planning to make supplemental disclosure or amendment related to the merger, in response to the lawsuit.

If the supplemental disclosure is specifically in response to a lawsuit, confirm that the lawsuit is the Target Case. Look for supporting details such as the plaintiff name or matching case details. If there is no evidence that the supplemental disclosure is associated with the Target Case, you will answer "No" for "supplemental_disclosure".

Otherwise, you will answer "Yes" for "supplemental_disclosure" and proceed to the next step.

3. **Search for Explicit Settlement**: determine whether there is an explicit "settlement" for the Target Case. Settlement can be indicated by:

    a. explicit mention of a "Settlement Agreement". This indicates that the agreement type is "Settlement Agreement".
    b. IF there is no "Settlement Agreement" mention, search for "MOU" or "memorandum of understanding" related to the case. This indicates the agreement type to be "MOU".

Note that in our definition, plaintiffs or the counsels of plaintiffs voluntarily dismissing a case in response to supplemental disclosure does NOT suggest a settlement. "Settlement" or "MOU" must be explicitly mentioned.

Stop and proceed to step 6 if you find a solid settlement indicator. 

4. **Search for other Agreements**: If there is no "Settlement Agreement" or "MOU", search for an explicit mention of a formal "XXX agreement" that indicates a resolution. 

Note that plaintiff can voluntarily dismiss a case without an agreement, and thus the dismissal of the case without explicit reason does NOT suggest an agreement.

Also, "agreeing to make supplemental disclosure" is NOT considered an agreement for this task. The agreement must be a formal, named agreement type and the detailed resolution of the agreement should be more specific than just supplemental disclosure.

5. **Handle Implied Agreement**: If an agreement is **strongly** implied but does not explicitly use the phrase 'XXX Agreement', classify it as 'Yes (Implied Agreement)' and note the lack of explicit terminology in the "note" field with "Agreement is strongly implied but not explicit". Do not attempt to classify it into the specific type of agreement. Err on the side of caution and only use this classification when the implication is very clear.

6. **Confirm the agreement pertains the Target Case**: If the agreement is case-specific, check the context around the agreement discussion to confirm that it pertains to the target case. Think step by step:

Evaluate whether the agreement was case specific (e.g. the Company reached the agreement with plaintiffs in XXX action; the Company made supplemental disclosure in response to XXX action and XXX action.) If so, confirm that the associated lawsuit(s) was or contained the Target Case. Look for supporting details such as the plaintiff name or matching case details. 

When the agreement is case-specific, and that there is no evidence that the Target Case is associated with the discussed agreement, you will answer "No" for "reached_agreement". 

When the agreement was not case-specific:
- If there is no evidence suggesting the lawsuits are different, assume inclusion and classify accordingly. Add to the "note" field: "The identified agreement was not case specific."
- If any evidence suggests the lawsuits are different, classify as "No". 

If you find multiple agreement types, choose the most specific one. For example, if you find both "Settlement Agreement" and "MOU", choose "Settlement Agreement". If an MOU and another non-settlement agreement exist, prioritize "MOU".

If multiple "Other Agreements" exist, only answer with a single "Yes (Other Agreement)".

7. **Summarize the Results**: Gather all the evidence you have found and determine the final answer for "supplemental_disclosure" and "reached_agreement".

If step 2 finds any valid supplemental disclosure, you will answer "Yes" for "supplemental_disclosure". Otherwise, you will answer "No" for "supplemental_disclosure".

If Step 6 finds no valid agreement or the agreement explicitly pertains to a different case, answer 'No' for 'reached_agreement'. Otherwise, you will answer "Yes (xxx)" for "reached_agreement" where xxx is the agreement type (Settlement Agreement, MOU, Other Agreement, or Implied Agreement). 

#### Output Instructions

First, output your thinking process by following the Thinking Instructions step by step. For each step, restate the step in your own words and discuss your thinkings and findings. Start your output with "**Step 1: Confirm Case Discussion**".

Then, provide your final answer in a JSON object with the following format:

{
    "supplemental_disclosure": [STRING] "Yes" or "No",
    "supplemental_disclosure_quote": [STRING] exact quote from the filing that proved the existence of supplemental disclosure; "Not applicable" if there is no such discussion,
    "reached_agreement": [STRING] One of the following: "Yes (Settlement Agreement)", "Yes (MOU)", "Yes (Other Agreement)", "Yes (Implied Agreement)", or "No",
    "reached_agreement_quote": [STRING] exact quote from the filing that proved the "reached_agreement" result; "Not applicable" if there is no such discussion,
    "note": [STRING] Any other supportive comment or "Case not mentioned." if applicable
}

Never generate any comment or note within the JSON beyond the "outcome_details" or "note" fields. Finish your output with just the JSON object.
    ----------------
    EXCERPTS: <filing_excerpts>
Item 8. | Additional Information.

Item 8 of the Schedule 14D-9 is hereby amended and supplemented by inserting at the end of the subsection titled "Certain Litigation" the disclosure set forth below:

"On October 31, 2018, a purported stockholder filed a lawsuit in the Court of Chancery of the State of Delaware, captioned Joseph Pietras v. Richard D. Phillips, et al., Case No. 2018-0789 (Del. Ch.) (the "Pietras Chancery Complaint"). The Pietras Chancery Complaint names the Company Board, Sycamore Partners, Staples, Parent and Purchaser as defendants. The Pietras Chancery Complaint alleges that the Company Board breached its fiduciary duties and/or otherwise violated Delaware law (and that Sycamore Partners, Staples, Parent and Purchaser colluded in or aided and abetted in such violations) by, among other things, purportedly (i) failing to obtain the highest value available for the Company in the marketplace; (ii) initiating a process to sell the Company that purportedly undervalues the Company and vests the Company Board with benefits that are not shared equally by the Company's public stockholders; (iii) failing to provide in the Schedule 14D-9 all material information necessary for the Company's public stockholders to decide whether to tender their Shares in the Offer; and (iv) committing waste by approving the Contemplated Transactions. The Pietras Chancery Complaint seeks to enjoin the closing of the Offer unless and until the Company adopts a procedure to obtain a purportedly fairer agreement and discloses the requested information in the Schedule 14D-9 or, alternatively, to recover damages if the Offer closes without the adoption of such a procedure and the disclosure of such information. The Company believes that the action is without merit. The full complaint is attached hereto as Exhibit (a)(5)(T)."

Item 9. | Exhibits.

Item 9 of the Schedule 14D-9 is hereby amended and supplemented by adding the following exhibit:

--------------------------------------------------------------------------------------------------------------------------
| Exhibit No. | Description                                                                                              |
| (a)(5)(T)   | Complaint filed on October 31, 2018 (Joseph Pietras v. Richard D. Phillips, et al., Case No. 2018-0789). |
--------------------------------------------------------------------------------------------------------------------------

Signature. After due inquiry and to the best of my knowledge and belief, I certify that the information set forth in this statement is true, complete and correct.

-------------------------------------------------------------------------
| ESSENDANT INC.                                                        |
| By:            | /s/ Brendan McKeough                                 |
| Name:          | Brendan McKeough                                     |
| Title:         | Senior Vice President, General Counsel and Secretary |
-------------------------------------------------------------------------

Date: November 1, 2018
</filing_excerpts>

<target_case>
- **case number**: 2018-0789
- **case name**: Joseph Pietras v. Richard D. Phillips, et al., Case No. 2018-0789
- **case filing date**: 2018-10-31
- **case count**: Court of Chancery of the State of Delaware
</target_case>

    ----------------
    LLM RESPONSE: LLM Response: **Step 1: Confirm Case Discussion**

The first step is to confirm that there are discussions around the Target Case. The Target Case is "Joseph Pietras v. Richard D. Phillips, et al., Case No. 2018-0789". Upon reviewing the Filing Excerpts, I found that the case is indeed mentioned in the subsection titled "Certain Litigation" under Item 8. Therefore, I can confirm that the Target Case is discussed in the filing.

**Step 2: Search for Supplemental Disclosure**

The next step is to search for explicit mention of the Company agreeing or planning to make supplemental disclosure or amendment related to the merger, in response to the lawsuit. Upon reviewing the Filing Excerpts, I found that the Company has amended and supplemented Item 8 of the Schedule 14D-9 by inserting additional disclosure at the end of the subsection titled "Certain Litigation". This suggests that the Company has made supplemental disclosure in response to the lawsuit. However, I need to confirm that the lawsuit is the Target Case. The filing mentions that the Pietras Chancery Complaint was filed on October 31, 2018, which matches the case filing date of the Target Case. Therefore, I can confirm that the supplemental disclosure is associated with the Target Case.

**Step 3: Search for Explicit Settlement**

The next step is to determine whether there is an explicit "settlement" for the Target Case. Upon reviewing the Filing Excerpts, I did not find any explicit mention of a "Settlement Agreement" or "MOU" related to the case.

**Step 4: Search for other Agreements**

Since I did not find any explicit settlement, the next step is to search for an explicit mention of a formal "XXX agreement" that indicates a resolution. Upon reviewing the Filing Excerpts, I did not find any explicit mention of a formal agreement.

**Step 5: Handle Implied Agreement**

Since I did not find any explicit agreement, the next step is to determine if an agreement is strongly implied but not explicitly stated. Upon reviewing the Filing Excerpts, I did not find any strong implication of an agreement.

**Step 6: Confirm the agreement pertains the Target Case**

Since I did not find any agreement, this step is not applicable.

**Step 7: Summarize the Results**

Based on my findings, I can summarize the results as follows:

* The Company has made supplemental disclosure in response to the lawsuit, which is the Target Case.
* There is no explicit or implied agreement related to the Target Case.

**Final Answer**

{
    "supplemental_disclosure": "Yes",
    "supplemental_disclosure_quote": "Item 8 of the Schedule 14D-9 is hereby amended and supplemented by inserting at the end of the subsection titled 'Certain Litigation' the disclosure set forth below:",
    "reached_agreement": "No",
    "reached_agreement_quote": "Not applicable",
    "note": "Case mentioned and supplemental disclosure made in response to the lawsuit."
}"""},

    {
        "role": "assistant",
        "content": """Step 1: Confirm Case Discussion
The filing excerpts reference the same case—Joseph Pietras v. Richard D. Phillips, et al., Case No. 2018-0789, filed on October 31, 2018, in the Court of Chancery of the State of Delaware. The LLM correctly identifies this as the Target Case.

Step 2: Search for Supplemental Disclosure
The filing excerpts do mention that Item 8 of the Schedule 14D-9 was amended and supplemented. However, the LLM incorrectly states that this supplemental disclosure was made “in response” to the Pietras Chancery Complaint. The excerpts themselves do not specify that the supplemental disclosure was explicitly made in response to the lawsuit, only that the disclosure was updated in the section titled “Certain Litigation.”

Step 3: Search for Explicit Settlement
No mention of a Settlement Agreement or MOU appears in the excerpts. The LLM correctly notes the absence of an explicit settlement.

Step 4: Search for Other Agreements
No additional or formal agreements are described in the filing. This remains accurate.

Step 5: Handle Implied Agreement
There is no indication of any implied agreement in the filing excerpts. This assessment is correct.

Step 6: Confirm the Agreement Pertains to the Target Case
Because there is no identified agreement, determining whether one pertains to the Target Case is not applicable.

Step 7: Summarize the Results
While the LLM was correct in stating that the filing references the Target Case and that there is supplemental disclosure, it incorrectly concludes that the supplemental disclosure was made in response to the lawsuit. The filing’s language simply states that the disclosure was revised in the litigation section; it does not explicitly tie that revision to a response to the lawsuit.

Final Determination: The LLM did hallucinate by asserting that the supplemental disclosure was explicitly made in response to the lawsuit.

<answer>True</answer>
"""
    },

            {"role": "user", "content": f"""==== YOUR TURN ====
     TASK INSTRUCTIONS: {task_instruction}

    ----------------
    EXCERPTS: {excerpts}

    ----------------
    LLM RESPONSE: {llm_response}

REMEMBER TO END YOUR RESPONSE WITH <answer>True</answer> IF THE LLM HAS MADE A HALLUCINATION, AND <answer>False</answer> IF IT HAS NOT MADE A HALLUCINATION."""}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=MESSAGES
    ).choices[0].message.content

    return response

def inference_parallel(data_list: list, task_instruction: str, model: str = "gpt-4o", 
                      excerpts_key: str = 'excerpts', response_key: str = 'LLM_raw_response', 
                      max_workers: int = 10, log_file: str = "inference_results.txt") -> list:
    """
    Parallelizes the process of running inference on multiple examples.
    
    Args:
        data_list: List of dictionaries containing LLM responses and excerpts
        task_instruction: The instruction given for the task
        model: The OpenAI model to use
        excerpts_key: Key in the dictionaries for excerpts
        response_key: Key in the dictionaries for LLM responses
        max_workers: Maximum number of parallel workers
        log_file: File to save the inference results
    Returns:
        List of inference results in the same order as data_list
    """
    import concurrent.futures
    from openai import OpenAI
    
    client = OpenAI()
    results = [None] * len(data_list)  # Pre-allocate results list with same length as data_list
    
    def process_item(index, item):
        try:
            result = inference(
                excerpts=item.get(excerpts_key, ''),
                llm_response=item.get(response_key, ''),
                task_instruction=task_instruction,
                model=model,
                client=client
            )
            return index, result
        except Exception as exc:
            print(f'Generated an exception for item {index}: {exc}')
            return index, f"ERROR: {str(exc)}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks with their indices
        futures = [executor.submit(process_item, i, item) for i, item in enumerate(data_list)]
        
        # Process results as they complete, but store them in the correct position
        for future in concurrent.futures.as_completed(futures):
            try:
                index, result = future.result()
                results[index] = result  # Store result at the correct index
            except Exception as exc:
                print(f'Error retrieving result: {exc}')
    
    with open(log_file, "w") as f:
        for i, result in enumerate(results):
            docket_id = data_list[i].get('docket_id', f'Example {i+1}')
            f.write(f"Inference Result for {docket_id}:\n")
            f.write(result)
            f.write("\n\n" + "-"*80 + "\n\n")
    
    return results


def extract_answer(text: str) -> bool:
    """
    Extracts the answer value between <answer> and </answer> tags from a string.
    
    Args:
        text: String containing an answer enclosed in <answer> tags
        
    Returns:
        bool: True if the answer is "True", False if the answer is "False"
        
    Raises:
        ValueError: If the answer is not found or is not "True" or "False"
    """
    import re
    
    # Find the content between <answer> and </answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if not match:
        raise ValueError("No answer found between <answer> and </answer> tags")
    
    answer_text = match.group(1).strip()
    
    # Check if the answer is valid
    if answer_text.lower() == "true":
        return True
    elif answer_text.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid answer: '{answer_text}'. Answer must be 'True' or 'False'")



if __name__ == "__main__":
    ### CONFIG
    import os
    NUM_TRIALS = 3
    MODEL_NAME = "osiris"
    FILTER_TYPE = "correct_only"
    LOG_FILE = os.path.join(os.path.dirname(__file__), f"inference_results-{MODEL_NAME}-{FILTER_TYPE}-supplemental-disclosure.txt")

    # print(f"Running inference for {NUM_TRIALS} trials with model {MODEL_NAME} and filter type {FILTER_TYPE}")

    file_path = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/clh-ma-supplemental-disclosure.csv"
    task_instruction_file = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/prompts/supplemental_disclosure.txt"
    CSV_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), f"hallucination_results-{MODEL_NAME}-{FILTER_TYPE}-supplemental-disclosure.csv")
    with open(task_instruction_file, 'r') as file:
        task_instruction = file.read()


    row_data: List[dict] = extract_data(
        file_path,
        excerpts_col="content",
        llm_response_col="raw_response",
        label_col="supplemental_disclosure_correct",
        note_col="RZ_note",
        filter_type=FILTER_TYPE
    )
    print(f"Number of fetched incorrect judgments: {len(row_data)}")
    
    # Run inference to find cases where model response is incorrect but judge believes there's no hallucination
    data_examples = []
    for row in row_data:
        data_examples.append(Example(
            input=task_instruction,
            actual_output=row['LLM_raw_response'],
            retrieval_context=[row["excerpts"]],
            additional_metadata={
                "docket_id": row["docket_id"],
                "correct": row["correct"],
                "note": row["RZ_note"]
            }
        ))
    
    judgment_client = JudgmentClient()

    inference_results = judgment_client.run_evaluation(
        examples=data_examples,
        scorers=[GroundednessScorer(threshold=1.0)],
        model=MODEL_NAME,
        eval_run_name="supplemental_disclosure_halu",
        project_name="haludetect",
        override=True
    )

    hallucination_results = [not result.success for result in inference_results]

    # Count the number of True and False results
    true_count = hallucination_results.count(True)
    false_count = hallucination_results.count(False)
    
    # Print the counts
    print(f"Hallucination Results Summary:")
    print(f"  True: {true_count} ({true_count/len(hallucination_results)*100:.2f}%)")
    print(f"  False: {false_count} ({false_count/len(hallucination_results)*100:.2f}%)")
    print(f"  Total: {len(hallucination_results)}")

         
    # Write results to CSV
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = ['excerpts', 'llm_response', 'docket_id', 'correct', 'hallucination_decision', 'model_raw_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, (row, hallucination, result) in enumerate(zip(row_data, hallucination_results, inference_results)):
            writer.writerow({
                'excerpts': row['excerpts'],
                'llm_response': row['LLM_raw_response'],
                'docket_id': row['docket_id'],
                'correct': row['correct'],
                'hallucination_decision': 'hallucination' if hallucination else 'no hallucination',
                'model_raw_response': result.scorers_data[0].additional_metadata["raw_response"]
            })
    
    print(f"Results written to {CSV_OUTPUT_FILE}")