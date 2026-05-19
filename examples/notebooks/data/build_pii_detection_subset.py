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

"""Build pii_detection_subset.csv for the GLiNER PII notebook.

Each example is a synthetic insurance-claim narrative with hand-tagged PII entities.
All names, SSNs, phone numbers, addresses, VINs, etc. are fabricated.

Phone numbers follow the NANP fictional reservation: `(XXX) 555-01YY` with valid area codes
and line numbers in the 0100-0199 reserved range. Area codes match the row's city/state
where one is given; otherwise generic US-metro NPAs are used.

Coverage notes:
  - SSN format variants: with dashes (#2, #12), no dashes (#3)
  - Phone format variants: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX
  - Ambiguous tokens (vehicle make vs. last name): "Ford" (#6), "Ram" (#19)
  - Unicode names + non-US addresses: #13 (Björn Sundström — Swedish), #16 (Marie Tremblay / Québec)
  - Minimal-PII narratives for false-positive testing: #8, #9, #20
  - Two-person narratives: #11
"""

import csv
import json
from pathlib import Path

# Each example: (text, [(entity_type, surface_text), ...])
# Entity offsets are computed automatically from the first occurrence of surface_text in text.
EXAMPLES = [
    (
        "Claimant Mary Johnson reported a rear-end collision at the intersection of Oak Street and Elm Avenue on March 14. Her vehicle, a 2019 Toyota Camry (VIN 1HGCM82633A123456), sustained damage to the rear bumper. Contact: mary.johnson@example.com or (415) 555-0123.",
        [
            ("first_name", "Mary"),
            ("last_name", "Johnson"),
            ("vehicle_identifier", "1HGCM82633A123456"),
            ("email", "mary.johnson@example.com"),
            ("phone_number", "(415) 555-0123"),
        ],
    ),
    (
        "Policy holder William O'Brien filed for theft of his 2021 Ford F-150 on October 7. SSN 123-45-6789. Contact via 503.555.0124 or wobrien@example.org.",
        [
            ("first_name", "William"),
            ("last_name", "O'Brien"),
            ("ssn", "123-45-6789"),
            ("phone_number", "503.555.0124"),
            ("email", "wobrien@example.org"),
        ],
    ),
    (
        "Sarah Chen, DOB 04/12/1985, reports vandalism to her vehicle parked overnight at 1247 Pacific Coast Hwy, Santa Monica, CA 90401. SSN 987654321.",
        [
            ("first_name", "Sarah"),
            ("last_name", "Chen"),
            ("date_of_birth", "04/12/1985"),
            ("street_address", "1247 Pacific Coast Hwy"),
            ("city", "Santa Monica"),
            ("state", "CA"),
            ("postcode", "90401"),
            ("ssn", "987654321"),
        ],
    ),
    (
        "Insured: Marcus Williams. Primary phone (208) 555-0125. Cell 208-555-0126. Email m.williams@example.org. Mailing address: 88 Cedar Lane, Boise, ID 83702.",
        [
            ("first_name", "Marcus"),
            ("last_name", "Williams"),
            ("phone_number", "(208) 555-0125"),
            ("phone_number", "208-555-0126"),
            ("email", "m.williams@example.org"),
            ("street_address", "88 Cedar Lane"),
            ("city", "Boise"),
            ("state", "ID"),
            ("postcode", "83702"),
        ],
    ),
    (
        "Hit-and-run incident reported by Daniel Park (DOB 11/30/1972) at the corner of 5th Avenue and Main Street, Portland, OR 97201. Witness took photo of plate ABC-1234.",
        [
            ("first_name", "Daniel"),
            ("last_name", "Park"),
            ("date_of_birth", "11/30/1972"),
            ("city", "Portland"),
            ("state", "OR"),
            ("postcode", "97201"),
            ("license_plate", "ABC-1234"),
        ],
    ),
    (
        "Vehicle damage claim from Patricia Ford for her 2020 Ford Escape. Note: 'Ford' is her surname, not just the vehicle make. Reach her at p.ford@example.net.",
        [
            ("first_name", "Patricia"),
            ("last_name", "Ford"),
            ("email", "p.ford@example.net"),
        ],
    ),
    (
        "Single-vehicle accident: insured Robert Garcia struck a utility pole at 4421 Westwood Blvd, Los Angeles, CA 90024 on January 3. Reachable at (213) 555-0127.",
        [
            ("first_name", "Robert"),
            ("last_name", "Garcia"),
            ("street_address", "4421 Westwood Blvd"),
            ("city", "Los Angeles"),
            ("state", "CA"),
            ("postcode", "90024"),
            ("phone_number", "(213) 555-0127"),
        ],
    ),
    (
        "Hail damage report: 2018 Subaru Outback with multiple dents across the hood and roof. Estimated repair cost $2,400. No injuries.",
        [],  # FP test — no PII
    ),
    (
        "Animal collision incident: claimant struck a deer on Highway 101 northbound around 6:15 AM Tuesday. Vehicle drivable. Standard comprehensive claim.",
        [],  # FP test — no PII
    ),
    (
        "Stolen vehicle: 2022 Honda Civic, vehicle identification number 2HGFC2F59NH123456. Last seen 9 PM Friday at owner's residence. Owner: Elizabeth Tran.",
        [
            ("vehicle_identifier", "2HGFC2F59NH123456"),
            ("first_name", "Elizabeth"),
            ("last_name", "Tran"),
        ],
    ),
    (
        "Two-party rear-end collision. Insured: Michael Lee at 312-555-0128. Other driver: Jennifer Okafor, jen.okafor@example.com. No injuries either side.",
        [
            ("first_name", "Michael"),
            ("last_name", "Lee"),
            ("phone_number", "312-555-0128"),
            ("first_name", "Jennifer"),
            ("last_name", "Okafor"),
            ("email", "jen.okafor@example.com"),
        ],
    ),
    (
        "Parking lot scrape reported by Aisha Patel, Apt 4B, 215 W 42nd St, New York, NY 10036. SSN 555-66-7777.",
        [
            ("first_name", "Aisha"),
            ("last_name", "Patel"),
            ("street_address", "215 W 42nd St"),
            ("city", "New York"),
            ("state", "NY"),
            ("postcode", "10036"),
            ("ssn", "555-66-7777"),
        ],
    ),
    (
        "Vehicle theft claim: Björn Sundström, residing at 1900 Larkin St, San Francisco, CA 94109, phone (415) 555-0129.",
        [
            ("first_name", "Björn"),
            ("last_name", "Sundström"),
            ("street_address", "1900 Larkin St"),
            ("city", "San Francisco"),
            ("state", "CA"),
            ("postcode", "94109"),
            ("phone_number", "(415) 555-0129"),
        ],
    ),
    (
        "Insured reports the right rear door was struck by a runaway shopping cart in a grocery store parking lot Tuesday afternoon. Cosmetic damage only. The store has agreed to file a separate claim. Insured: Carlos Mendoza, c.mendoza@example.com.",
        [
            ("first_name", "Carlos"),
            ("last_name", "Mendoza"),
            ("email", "c.mendoza@example.com"),
        ],
    ),
    (
        "Claim from Sarah Williams-Brown for vehicle damage during a flood event. Address: 770 SE 7th Ave, Miami, FL 33131. Phone 305.555.0130.",
        [
            ("first_name", "Sarah"),
            ("last_name", "Williams-Brown"),
            ("street_address", "770 SE 7th Ave"),
            ("city", "Miami"),
            ("state", "FL"),
            ("postcode", "33131"),
            ("phone_number", "305.555.0130"),
        ],
    ),
    (
        "Cross-border auto claim: insured Marie Tremblay, residing in Montréal, Québec H3B 4G5. Vehicle damaged at 200 University Ave, Toronto, ON. Email: m.tremblay@example.ca.",
        [
            ("first_name", "Marie"),
            ("last_name", "Tremblay"),
            ("city", "Montréal"),
            ("state", "Québec"),
            ("postcode", "H3B 4G5"),
            ("street_address", "200 University Ave"),
            ("city", "Toronto"),
            ("state", "ON"),
            ("email", "m.tremblay@example.ca"),
        ],
    ),
    (
        "Contact policyholder David Kim at david.kim+claims@example.com or work line (206) 555-0131 ext 213. Personal cell: 206-555-0132.",
        [
            ("first_name", "David"),
            ("last_name", "Kim"),
            ("email", "david.kim+claims@example.com"),
            ("phone_number", "(206) 555-0131"),
            ("phone_number", "206-555-0132"),
        ],
    ),
    (
        "Commercial fleet claim from Henderson Logistics Inc. Fleet manager: Thomas Reeves, t.reeves@example-logistics.com, (469) 555-0133. Vehicle: 2023 Freightliner Cascadia, VIN 3AKJHHDR8NSXY12345.",
        [
            ("first_name", "Thomas"),
            ("last_name", "Reeves"),
            ("email", "t.reeves@example-logistics.com"),
            ("phone_number", "(469) 555-0133"),
            ("vehicle_identifier", "3AKJHHDR8NSXY12345"),
        ],
    ),
    (
        "Claimant Jordan Ramirez was struck while driving a 2021 Ram 1500. Damage to driver-side door. Contact: 720-555-0134.",
        [
            ("first_name", "Jordan"),
            ("last_name", "Ramirez"),
            ("phone_number", "720-555-0134"),
        ],
    ),
    (
        "Insured filed a comprehensive claim for tree-fall damage during a storm. No third parties involved. Vehicle parked in driveway at home address (on file). Claim documentation forthcoming.",
        [],  # FP test — references "home address" without exposing it
    ),
]


def find_nth(text: str, surface: str, occurrence: int) -> tuple[int, int]:
    """Find the nth (1-indexed) occurrence of surface in text; return (start, end)."""
    pos = -1
    for _ in range(occurrence):
        pos = text.find(surface, pos + 1)
        if pos == -1:
            raise ValueError(f"Occurrence {occurrence} of '{surface}' not found in: {text[:80]}...")
    return pos, pos + len(surface)


def build_entities(text: str, specs: list[tuple[str, str]]) -> list[dict]:
    """Resolve entity specs to {type, start, end, text} dicts, handling repeated surfaces.

    If the same (type, surface) pair appears twice, the second invocation matches the second
    occurrence in the text. Different types with the same surface text are independent.
    """
    seen_counts: dict[tuple[str, str], int] = {}
    entities = []
    for type_, surface in specs:
        key = (type_, surface)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        start, end = find_nth(text, surface, seen_counts[key])
        entities.append({"type": type_, "start": start, "end": end, "text": surface})
    return entities


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "pii_detection_subset.csv"

    rows = []
    for idx, (text, entity_specs) in enumerate(EXAMPLES, start=1):
        rows.append(
            {
                "example_id": f"pii_{idx:03d}",
                "text": text,
                "entities": json.dumps(build_entities(text, entity_specs), ensure_ascii=False),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "text", "entities"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Quick stats
    entity_count = sum(len(json.loads(r["entities"])) for r in rows)
    rows_no_pii = sum(1 for r in rows if json.loads(r["entities"]) == [])
    rows_with_pii = len(rows) - rows_no_pii
    print(f"Total entities: {entity_count}")
    print(f"Rows with no PII (FP-test rows): {rows_no_pii}")
    if rows_with_pii:
        print(f"Avg entities per PII-bearing row: {entity_count / rows_with_pii:.1f}")
    else:
        print("Avg entities per PII-bearing row: N/A (no PII-bearing rows in subset)")


if __name__ == "__main__":
    main()
