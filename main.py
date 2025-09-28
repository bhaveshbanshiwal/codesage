import streamlit as st
from gtts import gTTS
import io
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import pickle
from pydub import AudioSegment
from faster_whisper import WhisperModel
import subprocess
import os
import pickle
import threading
import time
import json
import random
from streamlit_ace import st_ace
import pandas as pd

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            order: 2;
        }
    </style>
    """,
    unsafe_allow_html=True
)

last = None

# f = open('gemini_api.pkl', 'rb')
# MY_API_KEY = pickle.load(f)
# f.close()
MY_API_KEY = 'AIzaSyDJnM1pk0B24R9dHGKgc_xHhoa8u4GCA-o'
genai.configure(api_key=MY_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ft = open('topics.bin', 'rb')
# topics = pickle.load(ft)
# ft.close()

f = open('exception_commands.bin', 'rb')
VULN_KEYS = pickle.load(f)
f.close()
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

def generate_performance_report(performance_data):
    if not performance_data:
        return "No performance data to analyze."

    report_prompt = "You are an expert programming coach. Analyze the following interview performance and provide a summary of the candidate's strengths and weaknesses based on the problem topics (tags). Be encouraging and provide actionable feedback.\n\nPerformance Summary:\n"
    for item in performance_data:
        tags = item.get('tags', [])
        if not isinstance(tags, list):
            tags = [] 
        report_prompt += f"- Problem ID {item['problem_id']} (Difficulty: {item['difficulty']}, Tags: {', '.join(tags)}): Scored {item['score']}/10\n"
    report_prompt += "\nAnalysis (Strengths and Weaknesses):"

    try:
        response = model.generate_content(report_prompt)
        return response.text
    except Exception as e:
        return f"Could not generate performance analysis. Error: {e}"

def compare_outputs(actual_str, expected_str):
    actual_str = actual_str.strip()
    expected_str = str(expected_str).strip()

    if actual_str.startswith('[') and actual_str.endswith(']') and \
       expected_str.startswith('[') and expected_str.endswith(']'):
        try:
            actual_list = json.loads(actual_str)
            expected_list = json.loads(expected_str)
            return sorted(actual_list) == sorted(expected_list)
        except (json.JSONDecodeError, TypeError):
            return actual_str == expected_str
    else:
        return actual_str == expected_str

def security_check_ifsafe(code, lang):
    for key in VULN_KEYS.get(lang, []):
        if key in code:
            return  False
    return True

def run_code(code, test_input=""):
    """Runs Python code with a given input via stdin."""
    if not security_check_ifsafe(code, 'python'):
        return "Error: Security check failed: Potentially unsafe code detected."
    try:
        with open("temp_script.py", "w") as f:
            f.write(code)
        process = subprocess.run(
            ["python", "temp_script.py"],
            input=test_input,
            capture_output=True, text=True, timeout=5
        )
        output = process.stdout.strip()
        errors = process.stderr.strip()
        return f"Error: {errors}" if errors else output.replace("'", '"')
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out."
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        if os.path.exists("temp_script.py"):
            os.remove("temp_script.py")

def run_cpp(code, test_input=""):
    """Compiles and runs C++ code with a given input."""
    if not security_check_ifsafe(code, 'cpp'):
        return "Error: Security check failed: Potentially unsafe code detected."
    try:
        with open("temp_script.cpp", "w") as f:
            f.write(code)
        compile_process = subprocess.run(
            ["g++", "temp_script.cpp", "-o", "temp_executable"],
            capture_output=True, text=True, timeout=10
        )
        if compile_process.returncode != 0:
            return f"Compilation Error: {compile_process.stderr.strip()}"
        
        run_process = subprocess.run(
            ["./temp_executable"],
            input=test_input,
            capture_output=True, text=True, timeout=5
        )
        output = run_process.stdout.strip()
        errors = run_process.stderr.strip()
        if errors:
            return f"Runtime Error: {errors}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out."
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        for f in ["temp_script.cpp", "temp_executable"]:
            if os.path.exists(f):
                os.remove(f)

def load_problems():
    """Loads problems from problems.json and groups them by difficulty."""
    try:
        with open('problems.json', 'r') as f:
            problems = json.load(f)
        difficulty_groups = {"Easy": [], "Medium": [], "Hard": []}
        for problem in problems:
            if problem.get('difficulty') in difficulty_groups:
                difficulty_groups[problem['difficulty']].append(problem)
        return difficulty_groups
    except (FileNotFoundError, json.JSONDecodeError):
        return {"Easy": [], "Medium": [], "Hard": []}

def transcribe_audio(audio_bytes):
    """Transcribes audio bytes to text using Whisper."""
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        segments, _ = whisper_model.transcribe("temp_audio.wav", beam_size=5)
        transcribed_text = "".join(segment.text for segment in segments)
        return transcribed_text.strip()
    except Exception as e:
        return f"Error during transcription: {e}"
    finally:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

def get_hint(problem_desc, user_code):
    """Generates a hint using the Gemini model."""
    hint_prompt = f"""
    You are an AI Tech Interviewer, Based on history of responses
        You are called for checking the code written by user and you may intervene if user is going wrong way.
    ---
    Problem: {problem_desc}
    ---
    Here is the student's current code attempt:
    ---
    Code: {user_code}
    ---
    History of responses:
    {st.session_state.messages}
    ---
    Your task is to provide a single, concise hint to help the student.
    - Ask Followup question if user is going wrong way even a little.
    - Do NOT give the full solution or large code blocks.
    - If the code has syntax errors, point towards the type of error.
    - If the code has a logical flaw, gently guide them to reconsider their approach for a specific part.
    - If the code is empty, suggest a good starting point or data structure to consider.
    - The hint should be encouraging and helpful.
    """
    try:
        response = model.generate_content(hint_prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't generate a hint at the moment. Error: {e}"


if "messages" not in st.session_state:
    st.session_state.messages = []
if "code_content" not in st.session_state:
    st.session_state.code_content = ""
if "problems" not in st.session_state:
    st.session_state.problems = load_problems()
if "current_problem" not in st.session_state:
    st.session_state.current_problem = None
if "test_results" not in st.session_state:
    st.session_state.test_results = None
if "level" not in st.session_state:
    st.session_state.level = 1
if "remaining_questions" not in st.session_state:
    st.session_state.remaining_questions = 3
if "score" not in st.session_state:
    st.session_state.score = []
if "performance_data" not in st.session_state: st.session_state.performance_data = []

st.markdown("Welcome, to the AI Technical Interviewer! Please select your preferred programming language (e.g., 'Python' or 'C++') to begin.")

if st.session_state.current_problem:
    problem = st.session_state.current_problem
    st.sidebar.header("Problem Statement")
    st.sidebar.markdown(f"**Difficulty: {problem.get('difficulty', 'N/A')}**")
    st.sidebar.markdown(problem.get('problemDescription', '...'))
    st.sidebar.divider()

if st.session_state.test_results:
    st.sidebar.header("Test Results")
    results = st.session_state.test_results
    for i, result in enumerate(results['details']):
        status = "✅ Passed" if result['correct'] else "❌ Failed"
        st.sidebar.markdown(f"**Test Case {i+1}: {status}**")
        with st.sidebar.expander("Details"):
            st.code(f"Input:\n{result['input']}\n\nYour Output:\n{result['output']}\n\nExpected Output:\n{result['expected']}", language='text')
    st.sidebar.subheader(f"Score: {results['score']}/10")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.remaining_questions <= 0:
    total_score = sum(st.session_state.score)
    st.success(f"Interview Over! Your total SCORE is {total_score}.")

st.session_state.lang_chosen = False

interview_over = st.session_state.remaining_questions <= 0
chat_placeholder = "Interview finished. Refresh to start a new one." if interview_over else "Your response..."
if interview_over:
    # one time run onyl
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = True
        st.session_state.current_problem = None # removr the final problem from the sidebar
        
        total_score = sum(st.session_state.score)
        st.session_state.messages.append({"role": "assistant", "content": f"Interview Over! Your total SCORE is {total_score}."})
        if st.session_state.performance_data:
            with st.spinner("Analyzing your performance..."):
                analysis_text = generate_performance_report(st.session_state.performance_data)
            st.session_state.messages.append({"role": "assistant", "content": analysis_text})
        
        st.rerun() 
        st.stop()

    if st.session_state.performance_data:
        st.header("Performance Chart")
        df = pd.DataFrame(st.session_state.performance_data)
        df['Question'] = df.apply(lambda row: f"ID {row['problem_id']} ({row['difficulty']})", axis=1)
        df.set_index('Question', inplace=True)
        st.bar_chart(df['score'])

if prompt := st.chat_input(chat_placeholder, disabled=interview_over):
    if ("yes" in prompt.lower() and "next" in prompt.lower()) or ('next' in prompt.lower()):
        prompt = "Yes, let's proceed to the next question."
        st.session_state.current_problem = None
        remaining = st.session_state.remaining_questions - 1
        st.session_state.remaining_questions = max(0, remaining)
        if remaining <= 0:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "You have completed all questions."})
            interview_over = True
            st.rerun()

    if not st.session_state.current_problem:
        st.session_state.messages.append({"role": "user", "content": prompt})
        difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}
        level_difficulty = difficulty_map.get(st.session_state.level, "Hard")
        available_problems = st.session_state.problems.get(level_difficulty, [])
        if available_problems:
            st.session_state.current_problem = random.choice(available_problems)
            st.session_state.lang_chosen = True
            st.session_state.test_results = None
            st.session_state.code_content = st.session_state.current_problem.get('starterCode', '')
            st.session_state.messages.append({"role": "assistant", "content": f"Great! Let's start with a {level_difficulty} problem. Please use the code editor in the sidebar."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any problems for your difficulty level."})
        st.rerun()

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "Please use the code editor on the right to solve the problem. If you need help, you can click 'Get a Hint'."})
        st.rerun()


if st.session_state.test_results:
    st.sidebar.header("Test Results")
    results = st.session_state.test_results
    for i, result in enumerate(results['details']):
        status = "✅ Passed" if result['correct'] else "❌ Failed"
        st.sidebar.markdown(f"**Test Case {i+1}: {status}**")
        with st.sidebar.expander("Details"):
            st.code(f"Input:\n{result['input']}\n\nYour Output:\n{result['output']}\n\nExpected Output:\n{result['expected']}", language='text')
    st.sidebar.subheader(f"Score: {results['score']}/10")

if st.session_state.current_problem:

    st.header("Your Solution")
    editor_language = 'python' 
    if len(st.session_state.messages) > 1:

        if st.session_state.messages:
            lang_choice = st.session_state.messages[0]['content'].lower()
            if 'c++' in lang_choice or 'cpp' in lang_choice:
                editor_language = 'c_cpp'
    st.markdown("You choose to solve the problem in **" + editor_language.capitalize() + "**. Write your code below and click 'Submit Code' when you're ready.")
    code_input = st_ace(
        value=st.session_state.code_content,
        language=editor_language,
        theme="tomorrow_night_blue",
        height=400,
        key="ace_editor",
        auto_update=True, # instant state update on typing
        wrap=True

    )
    # Keeping edior content in sync
    st.session_state.code_content = code_input

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        submitted = st.button("Submit Code")
    with col2:
        hint_requested = st.button("Get a Hint")
    with col3:
        audio = mic_recorder(start_prompt="🎤 Dictate Code", stop_prompt="⏹️ Stop", key='recorder')

    if audio:
        with st.spinner("Transcribing..."):
            transcribed_text = transcribe_audio(audio['bytes'])
            if "Error" not in transcribed_text:
                st.session_state.code_content = transcribed_text
                st.rerun()
            else:
                st.error(transcribed_text)

    if hint_requested:
        with st.spinner("Thinking of a hint..."):
            hint = get_hint(st.session_state.current_problem['problemDescription'], code_input)
            st.session_state.messages.append({"role": "assistant", "content": hint})
            st.rerun()


    if submitted:
        with st.spinner("Running tests..."):
            lang = st.session_state.messages[0]['content'].lower()
            run_func = run_code if 'python' in lang else run_cpp
            
            if problem.get('testCases'):
                passed, total = 0, len(problem['testCases'])
                test_results_details = []
                for tc in problem['testCases']:
                    output = run_func(code_input, tc.get('input', ''))
                    correct = compare_outputs(output, tc.get('output', ''))
                    if correct: passed += 1

                    test_results_details.append({
                        'input': tc.get('input', ''), 'output': output,
                        'expected': tc.get('output', ''), 'correct': correct
                    })
                
                score = round((passed / total) * 10) if total > 0 else 0
                st.session_state.score.append(score)
                st.session_state.test_results = {'details': test_results_details, 'score': score}
                
                st.session_state.performance_data.append({
                    "problem_id": problem.get('id'),
                    "difficulty": problem.get('difficulty'),
                    "tags": problem.get('tags', []),
                    "score": score
                })
                st.session_state.remaining_questions -= 1
                result_msg = f"Finished tests. Passed: {passed}/{total}. Score: {score}."
                st.session_state.messages.append({"role": "assistant", "content": result_msg})
                
                if st.session_state.remaining_questions > 0:
                    st.session_state.messages.append({"role": "assistant", "content": "Next question..."})
                
                st.session_state.last_checked_code = ""
                st.rerun()


def monitoring_bot():
    while True:
        time.sleep(7) # Check every 7 seconds
        if st.session_state.current_problem is not None:
            current_problem = st.session_state.current_problem['problemDescription'] if st.session_state.current_problem else None
            code_content = st.session_state.code_content
            global last
            if last == None:
                last = code_content
            elif code_content != last:
                st.session_state.last_checked_code = code_content
                

                intervention_prompt = f"""
                You are an AI programming interviewer observing a candidate solve a problem in real-time.
                Problem: {current_problem['problemDescription']}
                Candidate's current code (may be incomplete): {code_content}

                Your role is to intervene ONLY if the candidate's core logic or approach is fundamentally wrong.
                - If you see a syntax error, really minor mistake, or a small bug, do NOT intervene, and if major syntax error then intervene ASAP.
                - Ignore if u feel syntax error is due to incomplete code.
                - If the approach is viable or it's too early to tell, respond with ONLY: NO_INTERVENTION
                - If the approach is clearly flawed, provide a short, guiding question to prompt them to rethink.
                - Do NOT correct syntax. Do NOT give code. Be subtle and encouraging.
                """
                
                intervention = model.generate_content(intervention_prompt)
                print("Intervention check:", intervention.text)

                if intervention and "NO_INTERVENTION" not in intervention:
                    print("Intervention needed:", intervention)
                    st.session_state.messages.append({"role": "assistant", "content": intervention.text})
                    st.session_state.intervention_message = intervention
        

if __name__ == '__main__':

    st.session_state.thread_started = True
    # daemon thread so that it stops when main app stops
    thread = threading.Thread(target=monitoring_bot, daemon=True)
    thread.start()
    print("🚀 Background thread started.")