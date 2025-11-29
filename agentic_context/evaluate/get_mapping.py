import re
import os


def extract_facts_dict(output, topic):
    results = []

    match_created_context = re.match(
        r"Created context '([^']+)' with \d+ facts?\. Stored facts: \[(.*)\]", output
    )
    if match_created_context:
        context_id = match_created_context.group(1)
        facts_str = match_created_context.group(2)
        facts = [f.strip().strip("'\"") for f in facts_str.split(",")]
        for fact in facts:
            results.append({"fact": fact, "context": context_id, "topic": topic})
        return results

    match_added = re.search(r"Added to (.+?): (.+)", output.strip())
    if match_added:
        context_id = match_added.group(1)
        new_fact = match_added.group(2).strip()
        results.append({"fact": new_fact, "context": context_id, "topic": topic})
        return results


    match_updated = re.match(r"Updated ([^:]+): (.+?) → (.+)", output)
    if match_updated:
        context_id = match_updated.group(1)
        new_fact = match_updated.group(3).strip()
        results.append({"fact": new_fact, "context": context_id, "topic": topic})
        return results

    match_added = re.match(r"Added to ([^:]+): (.+)", output)
    if match_added:
        context_id = match_added.group(1)
        new_fact = match_added.group(2).strip()
        results.append({"fact": new_fact, "context": context_id, "topic": topic})
        return results

    return results



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

