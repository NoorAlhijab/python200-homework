# ==========================
# Task 1: Setup and System Prompt
# ==========================

from unittest import result

from dotenv import load_dotenv
from openai import OpenAI
from typer import prompt

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content
# System prompt that sets up the model as a job application coach.
system_prompt = """ 

You are an AI job application coach helping career changers and job seekers 
improve their job application materials.

Your role is to help users:
- Rewrite and improve resume bullet points to highlight transferable skills.
- Create and refine cover letter drafts.
- Provide guidance on presenting their experience for new career opportunities.
- Ask helpful follow-up questions when more information is needed.

Behavior guidelines:
- Stay positive and encouraging, even when providing constructive feedback.
- Stay focused only on job application materials and career advice.
- Avoid giving advice unrelated to job applications, resumes, or cover letters.
- Do not invent information like work experience, skills, certifications, or achievements that the user has not provided.
- Only provide guidance based on the user's input.
- Always remind the user to review, personalize, and edit your output before submitting it anywhere.
- Acknowledge that you may not know specific industry norms or hiring expectations, and encourage the user to use their own judgment.

"""

# Example conversation
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Help me rewrite my resume bullet point: worked on data projects."}
]

# response = get_completion(messages)
# print("Raw response:")
# print(repr(response))

# ==========================
# Task 2: Bullet Point Rewriter
# ==========================

import json

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.

    Rewrite each resume bullet point below to be more specific,
    results-oriented, and compelling.
    Use strong action verbs.
    Do not invent facts that aren't implied by the original.

    Respond ONLY with valid JSON.
    Do NOT use markdown.
    Do NOT use code fences such as ```json.
    Do NOT include explanations or any other text.

    Return a JSON list where each item has:
    - "original"
    - "improved"

    ---BEGIN BULLET POINTS---

    {bullet_text}

    ---END BULLET POINTS---

    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    try:
        rewritten_bullets = json.loads(response)

    except json.JSONDecodeError:
        print("Could not parse JSON response. Raw response:")
        print(response)
        return []

    for item in rewritten_bullets:
        print(f"Original : {item['original']}")
        print(f"Improved: {item['improved']}\n")

    return rewritten_bullets

# These bullets are weak because they are too general
# The model improved them by using stronger action verbs and clearer wording

# ==========================
# Task 3: Cover Letter Generator
# ==========================

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    return response

# Test function
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)

# Chose these examples because they show how to connect previous experience
# to a new career while keeping the tone confident and specific.
# Few-shot prompting helps control the style, structure, and level of detail
# in the generated cover letter.

# ==========================
# Task 4: Moderation Check
# ==========================

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Your message was flagged. Please rephrase your input.")
        return False
    else:
        print("Message is safe.")
        return True

# Test safe input
safe_test = "Help me improve my resume for a Junior Data Analyst position."
print("Safe test result:", is_safe(safe_test))  # Expected: True    

# Test flagged input
flagged_test = "I want to hack into computer systems to steal data." # Expected to be flagged
print("Flagged test result:", is_safe(flagged_test))  # Expected: False

# ==========================
# Task 5: The Chatbot Loop
# ==========================

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            results = rewrite_bullets(raw_bullets)
            print("\nJob Application Helper: Here are your improved bullet points:\n")
            for item in results:
                print("Original : ", item['original'])
                print("Improved: ", item['improved'])
                print()
            messages.append({"role": "user", "content": f"Rewrite these resume bullet points:\n{'\n'.join(raw_bullets)}"})
            messages.append({"role": "assistant", "content": json.dumps(results)})
                
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            result = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:")
            print(result)
            messages.append({"role": "user","content": f"Create a cover letter for {job_title}. Background: {background}"})
            messages.append({"role": "assistant","content": result})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "user", "content": user_input})
            response = get_completion(messages)
            print("Job Application Helper:\n", response)
            messages.append({"role": "assistant", "content": response})
        
if __name__ == "__main__":
    run_chatbot()

# Chatbot memory test:
# I tested the chatbot by asking more than one question.
# It remembered the earlier conversation because the user and assistant
# messages are saved in the messages list.    

# ==========================
# Task 6: Ethics Reflection
# ==========================

# I chose the comment-block format for my ethics reflection.
#
# AI-generated job application advice may contain bias because the training data may not represent
# all industries, backgrounds, or communication styles equally. This could cause the chatbot to
# recommend certain resume formats or career advice that may not be suitable for every job seeker.
# Another limitation is that AI does not fully understand a person's complete career story, so users
# should review, personalize, and edit the generated content before submitting applications.
# Users should also avoid sharing sensitive personal information and use AI as a helpful tool rather
# than relying on it as the only source of career guidance.