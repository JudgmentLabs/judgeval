# Judgeval Java SDK

Java SDK for [Judgeval](https://judgmentlabs.ai/) - Agent Behavior Monitoring framework.

## Installation

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>com.judgmentlabs</groupId>
    <artifactId>judgeval-java</artifactId>
    <version>0.3.0</version>
</dependency>
```

## Setup

Set environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

[Create a free account](https://app.judgmentlabs.ai/register) to get your keys.

## Quick Start

```java
import com.judgmentlabs.judgeval.Judgeval;
import com.judgmentlabs.judgeval.data.Example;

public class SimpleExample {
    public static void main(String[] args) {
        var client = Judgeval.builder()
                .apiKey(System.getenv("JUDGMENT_API_KEY"))
                .organizationId(System.getenv("JUDGMENT_ORG_ID"))
                .build();
        
        var tracer = client.tracer()
                .create()
                .projectName("my_project")
                .build();
        
        tracer.span("agent.run", () -> {
            String result = "The capital is Paris";
            
            tracer.asyncEvaluate(
                client.scorers().builtIn().answerRelevancy().build(),
                Example.builder()
                    .property("input", "What is the capital of France?")
                    .property("actual_output", result)
                    .build()
            );
            
            return result;
        });
    }
}
```

## Documentation

- [Full Documentation](https://docs.judgmentlabs.ai/documentation)
- [Cookbooks](https://github.com/JudgmentLabs/judgment-cookbook)
- [Video Tutorials](https://www.youtube.com/@Alexshander-JL)

## Development

```bash
mvn clean test
mvn clean install
```

## Links

- [Judgment Cloud](https://app.judgmentlabs.ai/register)
- [Self-Hosting Guide](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)
- [GitHub](https://github.com/JudgmentLabs/judgeval)

