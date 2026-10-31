import asyncio
import json
import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tools import OPENAI_TOOLS, run_tool

load_dotenv()
client = AsyncOpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

MODEL = os.environ["LLM_MODEL"]
MAX_STEPS = 10
MIN_FETCHES = 2
MAX_CITATION_RETRIES = 2

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

SYSTEM = (
    "You are a research assistant. Follow this process:\n"
    "1. Call search at least once.\n"
    "2. Call fetch on at least 2 different URLs and read them. Never fetch the same URL twice.\n"
    "3. Only then write the report.\n"
    "Cover the WHOLE topic the user gave (performance, concurrency, ecosystem, "
    "hiring, developer productivity, real-world use), not just one angle.\n"
    "Cite every claim inline as [title](url). Only cite URLs that appeared in "
    "search results or that you fetched. If a claim has no source, leave it out. "
    "Never invent URLs."
)


def norm(url: str) -> str:
    return url.rstrip("/.,;")


def slugify(topic: str) -> str:
    """Turn a topic into a safe, readable file name."""
    name = re.sub(r"[^\w\s-]", "", topic).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "report"


def setup_logger(slug: str) -> logging.Logger:
    """Log to logs/<slug>.log (full detail) and to the console."""
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(f"agent.{slug}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(f"logs/{slug}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


async def run_agent(topic: str, logger: logging.Logger) -> str:
    logger.info(f"START topic={topic!r} model={MODEL}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Research this topic: {topic}"},
    ]
    claimed = set()      # fetch URLs already requested (prevents duplicates)
    fetched = set()      # pages successfully read
    seen_urls = set()    # URLs from search results + fetched pages
    citation_retries = 0
    force_final = False  # when True, tools are removed so the model must write
    dup_count = 0        # how many repeat-fetch requests we've blocked

    async with httpx.AsyncClient() as http_client:
        for step in range(1, MAX_STEPS + 1):
            kwargs = dict(model=MODEL, messages=messages, max_tokens=2000)
            if not force_final:
                kwargs["tools"] = OPENAI_TOOLS
            resp = await client.chat.completions.create(**kwargs)

            msg = resp.choices[0].message
            messages.append(msg)

            # ---- Model wants to finish: validate first ----
            if not msg.tool_calls:
                if len(fetched) < MIN_FETCHES and step < MAX_STEPS:
                    logger.info(
                        f"[step {step}] only {len(fetched)} page(s) fetched, sending model back"
                    )
                    messages.append({
                        "role": "user",
                        "content": f"You have only read {len(fetched)} page(s). "
                                   f"Fetch at least {MIN_FETCHES} sources before writing the report.",
                    })
                    continue

                cited = {norm(u) for u in URL_RE.findall(msg.content or "")}
                bad = sorted(u for u in cited if u not in seen_urls)
                if bad and citation_retries < MAX_CITATION_RETRIES and step < MAX_STEPS:
                    citation_retries += 1
                    logger.info(f"[step {step}] unverified citations {bad}, sending back")
                    messages.append({
                        "role": "user",
                        "content": "These URLs in your report never appeared in any search "
                                   "result or fetched page: " + ", ".join(bad) +
                                   ". Rewrite the report using only URLs from your research. "
                                   "Remove any claim you cannot source.",
                    })
                    continue

                if bad:
                    logger.info(f"[step {step}] WARNING: final report still has unverified URLs {bad}")
                logger.info(f"DONE in {step} step(s), {len(fetched)} page(s) read")
                return msg.content or ""

            # ---- Model requested tools: run them concurrently ----
            async def execute_call(call):
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    return call, {}, "Invalid tool arguments (not valid JSON). Try again."

                if name == "fetch":
                    key = norm(args.get("url", ""))
                    # check-and-claim happens before any await, so no race
                    if key in claimed:
                        return call, args, (
                            "You already fetched this page. Use a different source "
                            "or write the report."
                        )
                    claimed.add(key)

                output = await run_tool(name, args, http_client)

                if name == "fetch":
                    if output.startswith("Fetch failed"):
                        claimed.discard(key)  # allow a retry later
                    else:
                        fetched.add(key)
                        seen_urls.add(key)
                elif name == "search":
                    seen_urls.update(norm(u) for u in URL_RE.findall(output))
                return call, args, output

            results = await asyncio.gather(*(execute_call(c) for c in msg.tool_calls))

            # 1) Append ALL tool results first (they must follow the assistant message)
            for call, args, output in results:
                logger.info(
                    f"[step {step}] {call.function.name}({args}) -> {len(output)} chars"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                })

            # 2) THEN decide whether to force the final report (once per step)
            dup_count += sum(1 for _, _, o in results if o.startswith("You already fetched"))
            if len(fetched) >= MIN_FETCHES and (dup_count >= 2 or step >= MAX_STEPS - 2):
                force_final = True
                logger.info(f"[step {step}] forcing final report (tools disabled)")
                messages.append({
                    "role": "user",
                    "content": "Stop researching. Write the final report now, using only "
                               "pages you have already read. Cite sources as [title](url).",
                })

    logger.info("STOPPED: reached the maximum number of steps")
    return "Stopped: reached the maximum number of steps."