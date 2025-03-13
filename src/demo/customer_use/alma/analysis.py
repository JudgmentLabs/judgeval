from dotenv import load_dotenv
import os
import csv
from judgeval.data import Example
from judgeval import JudgmentClient
from judgeval.scorers import ComparisonScorer
from typing import List
import openai
import json
from concurrent.futures import ThreadPoolExecutor



def load_examples():
    """Load and parse the data from CSV file"""
    with open(os.path.join(os.path.dirname(__file__), "data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        data = list(reader)
    
    examples = []
    for row in data:
        id, draft_text, final_text = row
        example = Example(
            input=str(id),
            actual_output=str(draft_text),
            expected_output=str(final_text),
        )
        examples.append(example)
    return examples

def run_judgment_evaluation(examples: List[Example]):
    """
    Run evaluation using JudgmentClient
    
    Args:
        examples: List of Example objects
        
    Returns:
        List of boolean values indicating if the example is a false negative
    """
    client = JudgmentClient()
    scorer1 = ComparisonScorer(threshold=1.0, criteria="Professional Tone and Clarity", description="Maintain formal, respectful, and precise language for clear communication. Tailor the tone to suit the document's purpose and audience, enhancing professionalism and minimizing redundancy.")
    scorer2 = ComparisonScorer(threshold=1.0, criteria="Structure and Organization", description="Ensure logical content arrangement with a coherent flow and effective transitions. Highlight key points to enhance readability and maintain alignment with the document’s objectives.")
    scorer3 = ComparisonScorer(threshold=1.0, criteria="Achievements and Impact", description="Showcase leadership and transformative contributions using specific examples and metrics. Highlight professional influence, strengths, and innovative impacts.")
    scorer4 = ComparisonScorer(threshold=1.0, criteria="Use of Evidence", description="Employ concrete examples, credentials, and precise metrics to substantiate claims. Utilize industry-specific language to bolster narrative credibility.")

    # input = f"Dear Officer,\n\n\nI am writing to offer my strongest recommendation for Mr. Arjun Nair's visa petition as an individual of extraordinary ability in product management within the video gaming industry. Throughout his career, I have witnessed Mr. Nair's exceptional contributions, initially during his time at Playway and more recently in our collaboration at Catalyze Ventures (C-Vent), where he serves as a mentor.\n\n\nIn my role as a Partner at Catalyze Ventures, a leading venture capital firm, I focus on identifying and supporting innovative startups within the gaming and related sectors. My career background includes senior positions at Omniverse, such as Director of Product Management for the Virtual World, and I have accumulated substantial experience in the gaming industry. This includes serving as Vice President of Product Management at Indus Interactive, where I was responsible for the product strategies of major gaming franchises, and holding key leadership roles at Playway, a pioneer in social gaming.\n\n\nI first collaborated with Mr. Nair at Playway, where he advanced to the position of General Manager for Playway Poker. Playway, known for successful titles like GameFarm, WordPlay, and Playway Poker, has been a leader in mobile and social gaming since 2007. The company's acquisition by MegaGames Interactive in May 2022 for $12.7 billion stands as one of the largest deals in the gaming industry, underscoring the magnitude of its impact.\n\n\nDuring his time at Playway, Mr. Nair made significant contributions to the company's success. He joined as a Senior Product Manager and quickly advanced through the ranks, demonstrating his exceptional skills and leadership abilities. As General Manager of Playway Poker, Mr. Nair was responsible for the product vision, strategy, operations, and P&L of the $150M+ business across mobile and PC platforms. Under his leadership, the Playway Poker franchise expanded and solidified its dominance in the competitive online gaming market. He led a team of over 80 engineers, product managers, designers, producers, QA, and artists across New York, Austin, and Vancouver, driving sustained revenue growth and enhancing user engagement. Mr. Nair's efforts were instrumental in revitalizing the Playway Poker franchise. In 2017, Playway Poker experienced significant growth, with mobile revenue increasing by 78% year-over-year, largely due to enhancements to game features and operations. This success contributed to Playway's overall strong performance, as reported in various news articles and annual reports. The game's revitalization demonstrated the effectiveness of live operations and user engagement strategies, influencing Playway's broader portfolio management. Mr. Nair's leadership and strategic vision were key factors in achieving these impressive results. \n\n\nThroughout his tenure at Playway, Mr. Nair was recognized for his outstanding contributions and leadership. He was the recipient of several prestigious awards, including the \"Excellence Award,\" the highest honor available at Playway, awarded to one employee per quarter out of more than 1,500 employees. This award is given to employees who exemplify the company's spirit of connecting people through games and deliver results with a positive attitude and build the Playway community. Additionally, Mr. Nair received the \"Diamond Award in Product Management,\" awarded to one Product Manager across Playway per year. This accolade highlights his exceptional skills in product management and his ability to drive impactful results for the company.\n\n\nMr. Nair's ability to mentor and develop talent was also evident during his time at Playway. He played a pivotal role in shaping the company's product management culture and best practices, mentoring a new generation of product managers and fostering a thriving PM community within the organization. His dedication to excellence and his influence within the company were widely recognized and appreciated by his colleagues and leadership.\n\n\nAfter his successful tenure at Playway, Mr. Nair joined Horizon Interactive, a game developer with over 100 employees operating 15 games in active development across various genres, including collectible card games, massively multiplayer online games, and casual games. Horizon Interactive also operates ArcadeZone.com, an online gaming platform hosting a vast collection of games across different genres, offering both single-player and multiplayer experiences. As a founding partner and Chief Product Officer at Horizon Interactive, Mr. Nair has been instrumental in the company's growth and success. He drives the strategic vision and product development for the company, spearheads corporate development initiatives, and leads all game and product teams, ensuring the delivery of captivating and innovative gaming experiences to a diverse user base.\n\n\nOne of Mr. Nair's most notable achievements at Horizon Interactive has been his pivotal role in the company's major acquisitions. He played a key role in Horizon Interactive's acquisition of ArcadeZone, valued at over $100 million, and FunFrenzy, valued at over $5 million. Mr. Nair oversaw all phases of these acquisitions, from due diligence and valuation to integration and assessment, significantly enhancing the company's portfolio and market presence.\n\n\nIn addition to his contributions at Horizon Interactive, Mr. Arjun Nair has played a significant role in the wider gaming and tech industry as a venture scout for Catalyze Ventures (C-Vent). This esteemed venture capital firm is renowned for its rigorous selection process, choosing only those individuals and startups that display remarkable innovation and potential. Within this prestigious context, Mr. Nair stands out as a valuable asset, also serving as a mentor.\n\n\nIn his mentorship capacity, Mr. Nair is actively involved with the 2025 Q1 cohort of the SPEEDLANE program, C-Vent GAMES' exclusive accelerator for early-stage startups at the crossroads of technology and gaming. This highly selective program, admitting merely about 1% of applicants, provides substantial investment and connects participants with industry veterans and a network of driven founders. Through his mentorship, Mr. Nair imparts his extensive expertise in gaming, steering emerging companies toward success.\n\n\nApart from mentoring, Mr. Nair's role as a venture scout is crucial, as he evaluates and recommends innovative startups for C-Vent\u2019s portfolio. His keen eye for potential is demonstrated by his introduction of standout startups such as Glimmer Games, which has captivated a global audience with its popular casual games and attracted renowned investors. Another noteworthy introduction is NexTech, a company known for its cutting-edge AI-driven engagement solutions.\n\n\nMr. Nair's influence goes beyond his work with C-Vent, reaching into the broader venture capital landscape. He has served as a judge at an online demo day hosted by Launch Spectrum, where he meticulously assessed startups and provided valuable feedback. His strategic evaluation during this event, particularly recognizing NexTech's potential and facilitating its connection to C-Vent, underscores his dedication to fostering innovation and supporting the growth of early-stage companies. This role complements his responsibilities at C-Vent, highlighting the extensive reach of his engagement within the tech and gaming sectors.\n\n\n\n[Conclusion]\n\n\nIn conclusion, Mr. Nair's extensive experience and remarkable achievements in the gaming industry make him an invaluable asset to any organization. His strategic vision, leadership skills, and ability to drive significant business growth have been demonstrated time and again throughout his career. From his pivotal role in revitalizing the Playway Poker franchise to his contributions to Horizon Interactive's growth and success, Mr. Nair has consistently delivered outstanding results. His involvement in the venture capital ecosystem, both as a venture scout for C-Vent and a mentor for the SPEEDLANE program, further underscores his commitment to fostering innovation and supporting the next generation of tech entrepreneurs. Mr. Nair's unique blend of expertise, dedication, and passion for the gaming industry positions him as a leader who will continue to make significant contributions to the field.\n\n\nVery truly yours, \n\n\nEthan KimPartnerCatalyze Ventures (C-Vent)",
    # example = Example(
    #     input="jbdvb",
    #     actual_output=str(input),
    #     expected_output=str(input),
    # )
    output = client.run_evaluation(
        model="gpt-4o",
        examples=examples[:10],
        scorers=[scorer1, scorer2, scorer3, scorer4],
        eval_run_name="alma-basic-test2", 
        project_name="alma-basic-test3",
        override=True,
    )
    return output

def find_categories(examples: List[Example], current_categories: List[dict] = []):
    """
    Find the categories of the examples in parallel.
    """
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    def process_example(example):
        prompt = f"""
        You will be provided with a rough draft and a final version of a draft, where the final version is considered superior. Your task is to:

        1. Identify the differences between the two drafts based on the existing list of criteria: {current_categories}.
        3. If you find that a difference does not fit within this list, then you can either add a new criteria or update that criteria and description.
        4. Similarly, if criteria are similar or can be generalized, combine them and update the description to reflect the combined criteria.
        5. A sanity check is that the total number of criteria should not exceed 8; if it does, you should combine criteria that are most similar, again updating the description.

        Generate a JSON array of objects with the following format:
        [
            {{
                "criteria": "Criteria Name",
                "description": "Generic description of the criteria",
            }},
            ...
        ]

        Your response should include:
        - A detailed explanation of your reasoning.
        - The JSON array as specified. Ensure it is JSON formatted.

        Here are the drafts:
        Rough Draft: {example.actual_output}
        Final Version: {example.expected_output}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        try:
            response = response[response.index('json') + len('json'):].strip()
            response = response[response.index('['):response.rindex(']') + 1]
        except Exception as e:
            print(f"Error indexing JSON response: {response}, skipping example {example.input}")
            return example.input

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response}, skipping example {example.input}")
            return example.input

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_example, examples))

    # Filter out None results and combine valid results
    for result in results:
        if result:
            current_categories.extend(result)
        else:
            print("===================")
            print(f"Skipping example {result}")
            print("===================")

    return current_categories

def find_categories_in_batches(examples: List[Example]):
    """
    Process examples in batches of three and combine results.
    """
    current_categories = []
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    for i in range(0, len(examples), 3):
        batch = examples[i:i + 3]
        batch_categories = find_categories(batch, current_categories)
        
        # Combine the results from the batch into a single prompt
        combined_prompt = f"""
        You have processed the following batches of examples:
        {batch_categories}

        Please combine the results into a single coherent list of categories, ensuring that similar criteria are merged and descriptions are updated accordingly. Keep the formatting:
        A sanity check is that the total number of criteria should not exceed 6; if it does, you should combine criteria that are most similar, again updating the description.
        Also criteria should not be overly complex. Things like "Use of Evidence" and "Tone and Clarity" are good, but something like "Tone, Professionalism, Clarity, and Precision" is not.
        Ensure it is JSON formatted.
        [
            {{
                "criteria": "Criteria Name",
                "description": "description of the criteria",
            }},
            ...
        ]
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        ).choices[0].message.content

        try:
            response = response[response.index('json') + len('json'):].strip()
            response = response[response.index('['):response.rindex(']') + 1]
        except Exception as e:
            print(f"Error finding json for batch combination: {response}")
            continue

        try:
            current_categories = json.loads(response)
            print(f"Combined categories: {current_categories}")
            print()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response}")
            continue

    return current_categories

def main():
    load_dotenv()
    examples = load_examples()
    # categories = find_categories_in_batches(examples)

    # print("FINAL CATEGORIES")
    # print(categories)

    run_judgment_evaluation(examples)

if __name__ == "__main__":
    main()