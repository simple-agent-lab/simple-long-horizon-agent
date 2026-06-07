#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
with open('$WORKSPACE/learning_objectives.csv') as f:
    objs = list(csv.DictReader(f))
with open('$WORKSPACE/content_outline.json') as f:
    content = json.load(f)
questions = []
qid = 1
bloom_easy = ['Remember','Understand']
bloom_med = ['Apply','Analyze']
bloom_hard = ['Evaluate','Create']
templates = {
    'easy': [
        'What is the definition of {concept} in {topic}?',
        'List the key components of {concept}.',
        'Identify the main characteristics of {concept}.',
    ],
    'medium': [
        'How would you apply {concept} to solve a problem in {topic}?',
        'Compare and contrast {concept} with related concepts in {topic}.',
        'Explain how {concept} relates to other aspects of {topic}.',
    ],
    'hard': [
        'Evaluate the effectiveness of {concept} in the context of {topic}.',
        'Design a new approach using {concept} to address a challenge in {topic}.',
        'Critique the current understanding of {concept} in {topic}.',
    ],
}
answer_templates = {
    'easy': 'A factual description of {concept} covering its key properties and role in {topic}.',
    'medium': 'An analytical response demonstrating application of {concept} with examples from {topic}.',
    'hard': 'A critical evaluation of {concept} with evidence-based reasoning about {topic}.',
}
import random
random.seed(42)
for obj in objs:
    topic = obj['topic']
    concepts = content.get(topic, ['general concepts'])
    for diff in ['easy','medium','hard']:
        concept = random.choice(concepts)
        template = random.choice(templates[diff])
        q_text = template.format(concept=concept, topic=topic)
        a_text = answer_templates[diff].format(concept=concept, topic=topic)
        if diff == 'easy': bl = random.choice(bloom_easy)
        elif diff == 'medium': bl = random.choice(bloom_med)
        else: bl = random.choice(bloom_hard)
        questions.append({
            'id': f'Q{qid:03d}', 'objective_id': obj['objective_id'],
            'difficulty': diff, 'question': q_text, 'answer': a_text,
            'bloom_level': bl, 'topic': topic
        })
        qid += 1
from collections import Counter
by_diff = Counter(q['difficulty'] for q in questions)
by_topic = Counter(q['topic'] for q in questions)
stats = {
    'total_questions': len(questions),
    'by_difficulty': dict(by_diff),
    'by_topic': dict(by_topic),
    'objectives_covered': len(set(q['objective_id'] for q in questions))
}
json.dump({'questions': questions, 'statistics': stats},
          open('$WORKSPACE/question_bank.json','w'), indent=2)
"
