from judgeval import Tracer
from judgeval.v1.integrations.openlit import Openlit
from openai import OpenAI

tracer = Tracer.init(project_name="openlit")
Openlit.initialize(tracer)


openai = OpenAI()


@Tracer.observe()
def main():
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    main()
