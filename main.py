import asyncio
import sys
from pathlib import Path

from agent import run_agent, setup_logger, slugify


def main():
    topic = " ".join(sys.argv[1:]) or input("Topic: ")
    slug = slugify(topic)
    logger = setup_logger(slug)

    report = asyncio.run(run_agent(topic, logger))

    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports") / f"{slug}.txt"
    report_path.write_text(f"Topic: {topic}\n\n{report}\n", encoding="utf-8")

    print(f"\nReport saved to: {report_path}")
    print(f"Log saved to:    logs/{slug}.log")


if __name__ == "__main__":
    main()