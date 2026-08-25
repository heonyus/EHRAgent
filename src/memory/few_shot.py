###
# 처음에는 prompt_mimic.py 에서 정의한 프롬프트를 사용하여 few-shot을 파싱하고, 
# 이후에는 성공해서 memory.db에 적재된 데이터를 파싱하여 few-shot을 생성
###

import sqlite3
import Levenshtein

from pathlib import Path

DB_PATH = Path('data/memory.db')

def parse_shots(text):
    memory = []
    for item in text.split("\n\n"):
        item = item.split("Question:")[-1]
        question = item.split("\nKnowledge:\n")[0]
        item = item.split("\nKnowledge:\n")[-1]
        knowledge = item.split("\nSolution:")[0]
        code = item.split("\nSolution:")[-1]
        memory.append(
            {
                "question": question.strip(),
                "knowledge": knowledge.strip(),
                "code": code.strip(),
            }
        )
    return memory

def retrieve_examples(query, memory, num_shots=4):
    levenshtein_dist = {}
    for i in range(len(memory)):
        question = memory[i]["question"]
        levenshtein_dist[i] = Levenshtein.distance(query, question)
    levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
    selected_indexes = [levenshtein_dist[i][0] for i in range(min(num_shots, len(levenshtein_dist)))]
    examples = []
    for i in selected_indexes:
        template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
            memory[i]["question"],
            memory[i]["knowledge"],
            memory[i]["code"]
        )
        examples.append(template)
    return '\n'.join(examples)

def _connect(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            knowledge TEXT NOT NULL,
            code TEXT NOT NULL
        )
        """
    )
    return conn

def load_memory(db_path=DB_PATH):
    conn = _connect(db_path)
    rows = conn.execute('SELECT question, knowledge, code FROM examples ORDER BY id').fetchall()
    conn.close()
    return [{'question': row[0], 'knowledge': row[1], 'code': row[2]} for row in rows]

def append_memory(item, db_path=DB_PATH):
    conn = _connect(db_path)
    conn.execute('INSERT INTO examples (question, knowledge, code) VALUES (?, ?, ?)',
     (item['question'], item['knowledge'], item['code']))
    conn.commit()
    conn.close()

def seed_memory(text, db_path=DB_PATH):
    memory = load_memory(db_path)
    if memory:
        return memory
    for item in parse_shots(text):
        append_memory(item, db_path)
    return load_memory(db_path)
