from judgeval.judgment_client import JudgmentClient
from judgeval.data.example import Example
from judgeval.scorers import AnswerRelevancyScorer
from judgeval.common.tracer import Tracer

judgment = JudgmentClient()

# List of question-answer pairs
qa_pairs = [
    ("What is the capital of France?", "Paris"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    # ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
    # ("What is the chemical symbol for gold?", "Au"),
    # ("What is the square root of 144?", "12"),
    # ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    # ("What is the main component of the Sun?", "Hydrogen"),
    # ("What is the largest ocean on Earth?", "Pacific Ocean"),
    # ("Who discovered penicillin?", "Alexander Fleming"),
    # ("What is the capital of Japan?", "Tokyo"),
    # ("What is the hardest natural substance on Earth?", "Diamond"),
    # ("Who wrote 'To Kill a Mockingbird'?", "Harper Lee"),
    # ("What is the capital of Australia?", "Canberra"),
    # ("What is the largest mammal on Earth?", "Blue Whale"),
    # ("Who composed 'The Four Seasons'?", "Antonio Vivaldi"),
    # ("What is the capital of Brazil?", "Brasília"),
    # ("What is the chemical symbol for water?", "H2O"),
    # ("Who wrote 'The Great Gatsby'?", "F. Scott Fitzgerald"),
    # ("What is the capital of Egypt?", "Cairo"),
    # ("What is the largest desert in the world?", "Sahara Desert"),
    # ("Who painted 'The Starry Night'?", "Vincent van Gogh"),
    # ("What is the capital of India?", "New Delhi"),
    # ("What is the chemical symbol for iron?", "Fe"),
    # ("Who wrote '1984'?", "George Orwell"),
    # ("What is the capital of Canada?", "Ottawa"),
    # ("What is the largest bird in the world?", "Ostrich"),
    # ("Who composed 'Moonlight Sonata'?", "Ludwig van Beethoven"),
    # ("What is the capital of South Africa?", "Pretoria"),
    # ("What is the chemical symbol for silver?", "Ag"),
    # ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    # ("What is the capital of China?", "Beijing"),
    # ("What is the largest fish in the world?", "Whale Shark"),
    # ("Who painted 'The Last Supper'?", "Leonardo da Vinci"),
    # ("What is the capital of Russia?", "Moscow"),
    # ("What is the chemical symbol for oxygen?", "O"),
    # ("Who wrote 'The Catcher in the Rye'?", "J.D. Salinger"),
    # ("What is the capital of Germany?", "Berlin"),
    # ("What is the largest reptile in the world?", "Saltwater Crocodile"),
    # ("Who composed 'Symphony No. 9'?", "Ludwig van Beethoven"),
    # ("What is the capital of Italy?", "Rome"),
    # ("What is the chemical symbol for carbon?", "C"),
    # ("Who wrote 'The Lord of the Rings'?", "J.R.R. Tolkien"),
    # ("What is the capital of Spain?", "Madrid"),
    # ("What is the largest insect in the world?", "Goliath Beetle"),
    # ("Who painted 'The Scream'?", "Edvard Munch"),
    # ("What is the capital of Mexico?", "Mexico City"),
    # ("What is the chemical symbol for nitrogen?", "N"),
    # ("Who wrote 'The Hobbit'?", "J.R.R. Tolkien"),
    # ("What is the capital of South Korea?", "Seoul"),
    # ("What is the largest amphibian in the world?", "Chinese Giant Salamander")
]

# Create a list of Example objects
examples = [Example(input=question, actual_output=answer) for question, answer in qa_pairs]
for example in examples:
    print(example.model_dump())
judgment.run_evaluation(
    examples=examples,
    scorers=[AnswerRelevancyScorer(threshold=0.5)],
    append=True
)