import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRIPLET_PATTERN = re.compile(
    r"\(\s*([^,\n\r()]+?)\s*,\s*([^,\n\r()]+?)\s*,\s*([^\n\r()]+?)\s*\)"
)


@dataclass
class ProviderSpec:
    name: str
    provider_type: str  # "openai" or "gemini"
    model: str
    api_key_env: str
    api_base: Optional[str] = None


DEFAULT_PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "gpt4o": ProviderSpec(
        name="gpt4o",
        provider_type="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        api_base="https://api.openai.com/v1",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        provider_type="openai",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        api_base="https://api.deepseek.com/v1",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        provider_type="gemini",
        model="gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct organ-centric graph with multi-LLM synergy."
    )
    parser.add_argument(
        "--entity-file",
        type=str,
        default="graph/lab_organ_abnormality.md",
        help="Path to markdown file containing lab tests, organs, abnormalities.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graph/organ_graph.json",
        help="Path to output graph json.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="gpt4o,deepseek,gemini",
        help="Comma-separated providers from: gpt4o, deepseek, gemini.",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="gpt4o",
        help="Provider key used for LLM aggregation. Set empty to disable.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-N organs/relations for each entity.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for all LLM calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count for API calls.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output and skip completed entities.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between entities to avoid rate-limit bursts.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="graph/graph_construction_cache.json",
        help="Path to cache raw LLM responses.",
    )
    return parser.parse_args()


def parse_entity_markdown(path: Path) -> Tuple[List[str], List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Entity definition file not found: {path}")

    section = None
    labs: List[str] = []
    organs: List[str] = []
    abnormalities: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower == "# lab test":
            section = "lab"
            continue
        if lower == "# organ":
            section = "organ"
            continue
        if lower == "# imaging abnormality":
            section = "abnormality"
            continue
        if line.startswith("#"):
            section = None
            continue

        if section == "lab":
            labs.append(line)
        elif section == "organ":
            organs.append(line)
        elif section == "abnormality":
            abnormalities.append(line)

    if not labs or not organs:
        raise ValueError("Failed to parse labs/organs from markdown entity file.")

    return labs, organs, abnormalities


def build_entity_prompt(
    entity_name: str,
    entity_type: str,
    organ_names: Sequence[str],
    max_num_updates: int,
) -> str:
    if entity_type == "lab":
        concept_desc = f"laboratory test {entity_name}"
    elif entity_type == "abnormality":
        concept_desc = f"imaging abnormality {entity_name}"
    else:
        concept_desc = f"clinical concept {entity_name}"

    organ_list_str = ", ".join(organ_names)
    return (
        f"Given a {concept_desc}, extrapolate the relationships of it with the following organs "
        f"[{organ_list_str}] and provide a list of [{max_num_updates}] updates:\n"
        "1. Each update should be exactly in the format of (ENTITY 1, RELATIONSHIP, ENTITY 2).\n"
        "2. Both ENTITY 1 and ENTITY 2 should be nouns.\n"
        "3. The relationship is directed and describes a specific functional connection.\n"
        "4. Any element in (ENTITY 1, RELATIONSHIP, ENTITY 2) should be conclusive, and make it as short as possible.\n"
        "5. ENTITY 1 must be exactly the given concept name.\n"
        "6. ENTITY 2 must be one organ from the provided list.\n"
        "Return only the updates, one per line, no extra explanation."
    )


def build_aggregation_prompt(
    entity_name: str,
    entity_type: str,
    organ_names: Sequence[str],
    candidate_triplets_by_model: Dict[str, List[Tuple[str, str, str]]],
    max_num_updates: int,
) -> str:
    candidate_lines: List[str] = []
    for model_name, triplets in candidate_triplets_by_model.items():
        candidate_lines.append(f"- {model_name}:")
        if not triplets:
            candidate_lines.append("  (no valid triplets)")
            continue
        for h, r, t in triplets:
            candidate_lines.append(f"  ({h}, {r}, {t})")

    entity_desc = "laboratory test" if entity_type == "lab" else "imaging abnormality"
    organ_list = ", ".join(organ_names)
    candidates = "\n".join(candidate_lines)

    return (
        "You are aggregating multi-LLM outputs for organ-centric medical graph construction.\n"
        f"Target {entity_desc}: {entity_name}\n"
        f"Allowed organs: [{organ_list}]\n"
        f"Select top-{max_num_updates} most clinically plausible and consistent updates from candidates below.\n"
        "Rules:\n"
        "1. Use exactly format (ENTITY 1, RELATIONSHIP, ENTITY 2).\n"
        "2. ENTITY 1 must be exactly target concept name.\n"
        "3. ENTITY 2 must be from the allowed organs.\n"
        "4. Keep relations concise and clinically meaningful.\n"
        "5. Prefer consensus across models when possible.\n"
        "6. Output only the final updates, one per line.\n\n"
        f"Candidates:\n{candidates}"
    )


def _http_json_post(
    url: str,
    payload: Dict,
    headers: Dict[str, str],
    timeout: int,
) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai_compatible(
    spec: ProviderSpec,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> str:
    api_key = os.getenv(spec.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env: {spec.api_key_env}")
    if not spec.api_base:
        raise RuntimeError(f"Missing api_base for provider: {spec.name}")

    url = f"{spec.api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": spec.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical knowledge extraction assistant. "
                    "Follow output format strictly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = _http_json_post(url, payload, headers, timeout)
            return result["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError) as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"OpenAI-compatible call failed for {spec.name}: {last_err}")


def call_gemini(
    spec: ProviderSpec,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> str:
    api_key = os.getenv(spec.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env: {spec.api_key_env}")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model}"
        f":generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    headers = {"Content-Type": "application/json"}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = _http_json_post(url, payload, headers, timeout)
            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates returned: {result}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()
            if not text:
                raise RuntimeError(f"Empty text response: {result}")
            return text
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"Gemini call failed for {spec.name}: {last_err}")


def call_provider(
    spec: ProviderSpec,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> str:
    if spec.provider_type == "openai":
        return call_openai_compatible(spec, prompt, temperature, timeout, max_retries)
    if spec.provider_type == "gemini":
        return call_gemini(spec, prompt, temperature, timeout, max_retries)
    raise ValueError(f"Unsupported provider_type: {spec.provider_type}")


def normalize_name_case(name: str, canonical_names: Sequence[str]) -> Optional[str]:
    lower_to_canonical = {x.lower(): x for x in canonical_names}
    return lower_to_canonical.get(name.strip().lower())


def parse_triplets_from_text(
    text: str,
    expected_entity: str,
    allowed_organs: Sequence[str],
    top_n: int,
) -> List[Tuple[str, str, str]]:
    organ_lower_map = {o.lower(): o for o in allowed_organs}
    expected_lower = expected_entity.lower()
    parsed: List[Tuple[str, str, str]] = []
    seen = set()

    for m in TRIPLET_PATTERN.finditer(text):
        head = m.group(1).strip().strip("'\"")
        relation = m.group(2).strip().strip("'\"")
        tail = m.group(3).strip().strip("'\"")

        if head.lower() != expected_lower:
            continue
        canonical_tail = organ_lower_map.get(tail.lower())
        if canonical_tail is None:
            continue
        if not relation:
            continue
        key = (expected_entity, relation.lower(), canonical_tail.lower())
        if key in seen:
            continue
        seen.add(key)
        parsed.append((expected_entity, relation, canonical_tail))
        if len(parsed) >= top_n:
            break

    return parsed


def aggregate_by_consensus(
    entity_name: str,
    candidate_triplets_by_model: Dict[str, List[Tuple[str, str, str]]],
    top_n: int,
) -> List[Tuple[str, str, str]]:
    organ_to_relations: Dict[str, List[str]] = defaultdict(list)
    organ_votes: Counter = Counter()

    for triplets in candidate_triplets_by_model.values():
        voted_organs = set()
        for _, relation, organ in triplets:
            organ_to_relations[organ].append(relation)
            voted_organs.add(organ)
        for organ in voted_organs:
            organ_votes[organ] += 1

    ranked_organs = sorted(
        organ_votes.items(),
        key=lambda x: (-x[1], x[0]),
    )

    final_triplets: List[Tuple[str, str, str]] = []
    for organ, _ in ranked_organs[:top_n]:
        relation_counter = Counter(organ_to_relations.get(organ, []))
        if relation_counter:
            relation = relation_counter.most_common(1)[0][0]
        else:
            relation = "reflects the function of"
        final_triplets.append((entity_name, relation, organ))

    return final_triplets


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_enabled_provider_specs(provider_keys: Iterable[str]) -> Dict[str, ProviderSpec]:
    specs: Dict[str, ProviderSpec] = {}
    for key in provider_keys:
        k = key.strip()
        if not k:
            continue
        if k not in DEFAULT_PROVIDER_SPECS:
            raise ValueError(
                f"Unknown provider key: {k}. Available: {', '.join(DEFAULT_PROVIDER_SPECS.keys())}"
            )
        specs[k] = DEFAULT_PROVIDER_SPECS[k]
    return specs


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    entity_file = (root / args.entity_file).resolve()
    output_path = (root / args.output).resolve()
    cache_path = (root / args.cache_file).resolve()

    labs, organs, abnormalities = parse_entity_markdown(entity_file)
    entities: List[Tuple[str, str]] = [(x, "lab") for x in labs] + [
        (x, "abnormality") for x in abnormalities
    ]

    provider_specs = get_enabled_provider_specs(args.providers.split(","))
    if not provider_specs:
        raise ValueError("No providers enabled.")

    aggregator_key = args.aggregator.strip() if args.aggregator else ""
    use_llm_aggregation = bool(aggregator_key)
    if use_llm_aggregation and aggregator_key not in provider_specs:
        raise ValueError(
            f"Aggregator provider `{aggregator_key}` must be included in --providers."
        )

    existing_output: List[Dict] = load_json(output_path, default=[]) if args.resume else []
    completed_entity_to_entry = {x.get("id"): x for x in existing_output if "id" in x}
    cache = load_json(cache_path, default={})

    output_entries: List[Dict] = []
    if args.resume:
        output_entries.extend(existing_output)

    for idx, (entity_name, entity_type) in enumerate(entities, start=1):
        if args.resume and entity_name in completed_entity_to_entry:
            print(f"[{idx}/{len(entities)}] Skip existing: {entity_name}")
            continue

        prompt = build_entity_prompt(
            entity_name=entity_name,
            entity_type=entity_type,
            organ_names=organs,
            max_num_updates=args.top_n,
        )

        print(f"[{idx}/{len(entities)}] Building graph edges for: {entity_name}")
        candidate_triplets_by_model: Dict[str, List[Tuple[str, str, str]]] = {}
        raw_outputs_by_model: Dict[str, str] = {}

        for provider_key, spec in provider_specs.items():
            try:
                model_output = call_provider(
                    spec=spec,
                    prompt=prompt,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                )
            except Exception as err:
                print(f"  - {provider_key}: FAILED ({err})")
                model_output = ""

            parsed = parse_triplets_from_text(
                text=model_output,
                expected_entity=entity_name,
                allowed_organs=organs,
                top_n=args.top_n,
            )
            candidate_triplets_by_model[provider_key] = parsed
            raw_outputs_by_model[provider_key] = model_output
            print(f"  - {provider_key}: {len(parsed)} valid triplets")

        final_triplets: List[Tuple[str, str, str]] = []

        if use_llm_aggregation:
            agg_prompt = build_aggregation_prompt(
                entity_name=entity_name,
                entity_type=entity_type,
                organ_names=organs,
                candidate_triplets_by_model=candidate_triplets_by_model,
                max_num_updates=args.top_n,
            )
            try:
                agg_spec = provider_specs[aggregator_key]
                agg_output = call_provider(
                    spec=agg_spec,
                    prompt=agg_prompt,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                )
                final_triplets = parse_triplets_from_text(
                    text=agg_output,
                    expected_entity=entity_name,
                    allowed_organs=organs,
                    top_n=args.top_n,
                )
                raw_outputs_by_model[f"aggregator::{aggregator_key}"] = agg_output
            except Exception as err:
                print(f"  - aggregator {aggregator_key}: FAILED ({err}), fallback to consensus.")

        if not final_triplets:
            final_triplets = aggregate_by_consensus(
                entity_name=entity_name,
                candidate_triplets_by_model=candidate_triplets_by_model,
                top_n=args.top_n,
            )

        if not final_triplets:
            fallback_organ = normalize_name_case("blood", organs) or organs[0]
            final_triplets = [
                (entity_name, "reflects the functional status of", fallback_organ)
            ]

        output_entries.append(
            {
                "id": entity_name,
                "relation": [list(x) for x in final_triplets],
            }
        )

        cache[entity_name] = {
            "entity_type": entity_type,
            "providers": raw_outputs_by_model,
            "parsed_candidates": {
                k: [list(x) for x in v] for k, v in candidate_triplets_by_model.items()
            },
            "final_triplets": [list(x) for x in final_triplets],
        }

        save_json(output_path, output_entries)
        save_json(cache_path, cache)
        time.sleep(max(args.sleep, 0.0))

    print(f"Graph construction completed. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
