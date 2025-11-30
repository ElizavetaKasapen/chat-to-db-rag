import re
import os


# def extract_facts_dict(output, topic):
#     results = []

#     match_created_context = re.match(
#         r"Created context '([^']+)' with \d+ facts?\. Stored facts: \[(.*)\]", output
#     )
#     if match_created_context:
#         context_id = match_created_context.group(1)
#         facts_str = match_created_context.group(2)
#         facts = [f.strip().strip("'\"") for f in facts_str.split(",")]
#         for fact in facts:
#             results.append({"fact": fact, "context": context_id, "topic": topic})
#         return results

#     match_added = re.search(r"Added to (.+?): (.+)", output.strip())
#     if match_added:
#         context_id = match_added.group(1)
#         new_fact = match_added.group(2).strip()
#         results.append({"fact": new_fact, "context": context_id, "topic": topic})
#         return results


#     match_updated = re.match(r"Updated ([^:]+): (.+?) → (.+)", output)
#     if match_updated:
#         context_id = match_updated.group(1)
#         new_fact = match_updated.group(3).strip()
#         results.append({"fact": new_fact, "context": context_id, "topic": topic})
#         return results

#     match_added = re.match(r"Added to ([^:]+): (.+)", output)
#     if match_added:
#         context_id = match_added.group(1)
#         new_fact = match_added.group(2).strip()
#         results.append({"fact": new_fact, "context": context_id, "topic": topic})
#         return results

#     return results

import json
import os
import ast


def update_fact(results, fact_id, new_text):
    for fact in results:
        if fact["fact_id"] == fact_id:
            fact["fact"] = new_text
    return results

def extract_facts_dict(output, topic, facts_file_path="generated_facts.json"):
    """
    Extract facts from text output based on the new formats.
    If facts_file_path is provided, updates facts in the file for updated facts.
    """
    #results = []

    # Load existing facts if needed
    if facts_file_path and os.path.exists(facts_file_path):
        with open(facts_file_path, "r", encoding="utf-8") as f:
            facts_data = json.load(f)   # <-- list of facts
    else:
        facts_data = []
    
    # Process each line
    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Created context
        match_created = re.match(
            r"Created context '([^']+)' with \d+ facts\. Stored facts: (.+)",
            line
        )
        if match_created:
            context_id = match_created.group(1)
            raw_list = match_created.group(2).strip()

            try:
                fact_list = ast.literal_eval(raw_list)
            except Exception as e:
                print("ERROR parsing facts list:", e)
                continue

            for item in fact_list:
                fact_obj = {
                    "fact_id": item["id"],
                    "fact": item["text"],
                    "context": context_id,
                    "topic": topic
                }
                facts_data.append(fact_obj)

        # 2. Added fact
        match_added = re.match(
            r"In ([^ ]+) added new fact with fact_id: ([^\.]+)\. Fact text: (.+)\.",
            line
        )
        if match_added:
            context_id = match_added.group(1).strip()
            fact_id = match_added.group(2).strip()
            fact_text = match_added.group(3).strip()
            facts_data.append({
                "fact_id": fact_id,
                "fact": fact_text,
                "context": context_id,
                "topic": topic
            })
            continue

        # 3. Updated fact
        match_updated = re.match(
            r"Updated fact ([^ ]+) in ([^:]+): .+ → (.+)",
            line
        )
        if match_updated:
            fact_id = match_updated.group(1).strip()
            context_id = match_updated.group(2).strip()
            new_fact = match_updated.group(3).strip()
            facts_data = update_fact(facts_data, fact_id, new_fact)

            

    # Save back to file if path is provided
    if facts_file_path:
        with open(facts_file_path, "w", encoding="utf-8") as f:
            json.dump(facts_data, f, indent=2, ensure_ascii=False)

    #return results

#TODO put in separate file cause it's repeated twice
def load_text_files(root_dir):
    for folder, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(folder, filename)
                folder_name = os.path.basename(folder)
                file_name = filename

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        yield full_path, folder_name, file_name, f.read()
                except Exception as e:
                    print(f"Skipping {full_path}: {e}")


# response = """
# Created context 'Apples' with 2 facts. Stored facts: ['come in a variety of colors, the most common being red, green, and yellow', 'contain vitamins including vitamin C, thiamin, riboflavin, B6, vitamin A, and vitamin K']
# Added to Apples: are sweet and crisp
# Added to Apples: can be used in pies and juices
# Updated Apples: are sweet and crisp → are sweet, crisp, and juicy
# Created context 'Oranges' with 1 facts. Stored facts: ['are rich in vitamin C']
# Added to Oranges: have a tangy flavor
# """

# topic = "Fruits"

# # Call the function
# extracted_facts = extract_facts_dict(response, topic)

# # Print the results
# for f in extracted_facts:
#     print(f)