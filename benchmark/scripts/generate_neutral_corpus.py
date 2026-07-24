# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate a neutral text corpus for aiperf benchmarking by downloading
Wikipedia articles. Replaces the default shakespeare.txt corpus which
triggers excessive content-safety flags (~40-54%) for long-input scenarios.

Usage:
    python benchmark/scripts/generate_neutral_corpus.py \
        --output /path/to/corpus.txt \
        --target-mb 6
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Seed articles covering neutral, encyclopedic topics across science, technology,
# geography, history (non-violent), and nature. Avoids politics, crime, war, and
# any content likely to trigger content-safety flags.
SEED_TITLES = [
    # Science & Nature
    "Photosynthesis",
    "Cell (biology)",
    "Evolution",
    "Plate tectonics",
    "Water cycle",
    "Ecosystem",
    "Biodiversity",
    "Quantum mechanics",
    "General relativity",
    "Thermodynamics",
    "Electromagnetism",
    "Periodic table",
    "Organic chemistry",
    "Genetics",
    "Neuroscience",
    "Astronomy",
    "Solar System",
    "Milky Way",
    "Black hole",
    "Exoplanet",
    "Climate",
    "Oceanography",
    "Meteorology",
    "Geology",
    "Mineralogy",
    "Botany",
    "Zoology",
    "Microbiology",
    "Ecology",
    "Marine biology",
    # Technology & Engineering
    "Computer science",
    "Artificial intelligence",
    "Machine learning",
    "Internet",
    "Semiconductor",
    "Renewable energy",
    "Solar energy",
    "Wind power",
    "Hydroelectricity",
    "Nuclear power",
    "Electric vehicle",
    "Robotics",
    "Nanotechnology",
    "Biotechnology",
    "Telecommunications",
    "Satellite",
    "Global Positioning System",
    "Optical fiber",
    "3D printing",
    "Integrated circuit",
    # Geography & Nature
    "Amazon rainforest",
    "Sahara",
    "Himalaya",
    "Atlantic Ocean",
    "Pacific Ocean",
    "Mediterranean Sea",
    "Great Barrier Reef",
    "Amazon River",
    "Nile",
    "Yellowstone National Park",
    "Grand Canyon",
    "Great Wall of China",
    "Eiffel Tower",
    "Mount Everest",
    "Antarctica",
    "Arctic",
    "Galapagos Islands",
    # Mathematics
    "Mathematics",
    "Calculus",
    "Linear algebra",
    "Statistics",
    "Probability theory",
    "Number theory",
    "Geometry",
    "Topology",
    "Graph theory",
    "Cryptography",
    # History of Science & Ideas (non-violent)
    "Scientific Revolution",
    "Enlightenment",
    "Industrial Revolution",
    "Renaissance",
    "Ancient Greece",
    "Ancient Rome",
    "Ancient Egypt",
    "Silk Road",
    "Age of Exploration",
    "Agricultural revolution",
    # Culture & Society (neutral)
    "Music theory",
    "Architecture",
    "Painting",
    "Sculpture",
    "Literature",
    "Philosophy",
    "Linguistics",
    "Anthropology",
    "Archaeology",
    "Economics",
    "Psychology",
    "Sociology",
    "Education",
    "Library",
    "Museum",
    # Food & Health (neutral)
    "Nutrition",
    "Metabolism",
    "Human body",
    "Immune system",
    "Cardiovascular system",
    "Respiratory system",
    "Digestive system",
    "Medicine",
    "Pharmacology",
    "Public health",
    "Epidemiology",
]


def fetch_article(title: str) -> str:
    """Fetch plain text of a Wikipedia article via the API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "exsectionformat": "plain",
        "format": "json",
        "redirects": "true",
    }
    url = WIKIPEDIA_API + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeMoGuardrails-benchmark/1.0 (schilton@nvidia.com)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "") or ""
    except Exception as e:
        print(f"  Warning: failed to fetch '{title}': {e}", file=sys.stderr)
        return ""


def main():
    parser = argparse.ArgumentParser(description="Generate neutral Wikipedia corpus for aiperf")
    parser.add_argument("--output", default="benchmark/scripts/neutral_corpus.txt")
    parser.add_argument("--target-mb", type=float, default=6.0, help="Target corpus size in MB (default: 6)")
    args = parser.parse_args()

    target_bytes = int(args.target_mb * 1024 * 1024)
    output_path = args.output

    print(f"Target size: {args.target_mb} MB ({target_bytes:,} bytes)")
    print(f"Output: {output_path}")
    print(f"Fetching up to {len(SEED_TITLES)} Wikipedia articles...")

    collected = []
    total_bytes = 0

    for i, title in enumerate(SEED_TITLES):
        if total_bytes >= target_bytes:
            print(f"\nReached target size after {i} articles.")
            break
        print(f"  [{i + 1}/{len(SEED_TITLES)}] {title} ...", end=" ", flush=True)
        text = fetch_article(title)
        if text:
            # Strip excessive whitespace but keep paragraph breaks
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            collected.append(text)
            total_bytes += len(text.encode("utf-8"))
            print(f"{len(text):,} chars (total: {total_bytes / 1024 / 1024:.2f} MB)")
        else:
            print("skipped")
        time.sleep(0.1)  # be polite to Wikipedia API

    # If we haven't hit the target, cycle through articles again
    cycle = 0
    while total_bytes < target_bytes and cycle < 3:
        cycle += 1
        print(f"\nCycling through articles again (pass {cycle + 1}) to reach target size...")
        for text in list(collected):
            if total_bytes >= target_bytes:
                break
            collected.append(text)
            total_bytes += len(text.encode("utf-8"))

    corpus = "\n\n".join(collected)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    final_mb = len(corpus.encode("utf-8")) / 1024 / 1024
    print(f"\nWrote {final_mb:.2f} MB to {output_path}")
    print("Next step: copy to Brev node and replace shakespeare.txt:")
    print(f"  scp {output_path} ubuntu@brev-sax6j9j37:/tmp/neutral_corpus.txt")
    print(
        "  ssh ubuntu@brev-sax6j9j37 'cp /tmp/neutral_corpus.txt $(find /ephemeral/venv-aiperf -name shakespeare.txt)'"
    )


if __name__ == "__main__":
    main()
