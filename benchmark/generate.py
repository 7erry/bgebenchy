"""Generate a 100k Motion-like industrial catalog from the sample document shape."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Iterator

VENDORS = (
    "Schaeffler",
    "SKF",
    "Timken",
    "NSK",
    "NTN",
    "FAG",
    "INA",
    "McGill",
    "Baldor",
    "ABB",
    "SEW-Eurodrive",
    "Nord",
    "Gates",
    "Dodge",
    "Rexnord",
    "Boston Gear",
    "Martin",
    "Fenner",
    "Browning",
    "Lovejoy",
    "Dayco",
    "Regal Rexnord",
    "TB Woods",
    "Altra",
)

CATEGORIES: tuple[dict, ...] = (
    {
        "type": "Spherical Roller Bearings",
        "keyword": "Spherical Roller Bearing",
        "parent": "cat2000031",
        "parent_sf": "cat2000031-sf",
        "unspsc": "31171510",
        "pgc": "1110",
        "pgc_code": "ST.SM.SPHER.THRU SIZE 48",
        "cage": ("Roller Guided", "Window Type", "Machined Brass"),
        "series": ("222", "223", "230", "231", "232"),
        "unit": "bearing",
    },
    {
        "type": "Deep Groove Ball Bearings",
        "keyword": "Deep Groove Ball Bearing",
        "parent": "cat2000040",
        "parent_sf": "cat2000040-sf",
        "unspsc": "31171504",
        "pgc": "1104",
        "pgc_code": "ST.SM.BALL.DEEP GROOVE",
        "cage": ("Steel Cage", "Nylon Cage", "Brass Cage"),
        "series": ("6000", "6200", "6300", "6400"),
        "unit": "bearing",
    },
    {
        "type": "Tapered Roller Bearings",
        "keyword": "Tapered Roller Bearing",
        "parent": "cat2000034",
        "parent_sf": "cat2000034-sf",
        "unspsc": "31171512",
        "pgc": "1112",
        "pgc_code": "ST.SM.TAPER.SINGLE ROW",
        "cage": ("Stamped Steel", "Pin Type"),
        "series": ("302", "303", "313", "320"),
        "unit": "bearing",
    },
    {
        "type": "AC Motors",
        "keyword": "Three Phase AC Motor",
        "parent": "cat3000100",
        "parent_sf": "cat3000100-sf",
        "unspsc": "26101112",
        "pgc": "2101",
        "pgc_code": "EL.MT.AC.TEFC",
        "cage": ("Cast Iron", "Rolled Steel"),
        "series": ("EM", "CEM", "VEM", "ECP"),
        "unit": "motor",
    },
    {
        "type": "V-Belts",
        "keyword": "Classical V-Belt",
        "parent": "cat4000200",
        "parent_sf": "cat4000200-sf",
        "unspsc": "26111801",
        "pgc": "3205",
        "pgc_code": "PT.BL.V.CLASSICAL",
        "cage": ("Wrapped", "Raw Edge"),
        "series": ("A", "B", "C", "3V", "5V"),
        "unit": "belt",
    },
    {
        "type": "Roller Chain",
        "keyword": "ANSI Roller Chain",
        "parent": "cat4000210",
        "parent_sf": "cat4000210-sf",
        "unspsc": "26111503",
        "pgc": "3301",
        "pgc_code": "PT.CH.ROLLER.ANSI",
        "cage": ("Standard", "Heavy Series"),
        "series": ("40", "50", "60", "80", "100"),
        "unit": "chain",
    },
    {
        "type": "Jaw Couplings",
        "keyword": "Jaw Coupling",
        "parent": "cat4000220",
        "parent_sf": "cat4000220-sf",
        "unspsc": "26111506",
        "pgc": "3402",
        "pgc_code": "PT.CP.JAW",
        "cage": ("Nitrile Spider", "Urethane Spider"),
        "series": ("L050", "L070", "L095", "L110"),
        "unit": "coupling",
    },
    {
        "type": "Helical Gear Reducers",
        "keyword": "Helical Gear Reducer",
        "parent": "cat3000110",
        "parent_sf": "cat3000110-sf",
        "unspsc": "26111501",
        "pgc": "2503",
        "pgc_code": "PT.GR.HELICAL",
        "cage": ("Cast Iron Housing", "Aluminum Housing"),
        "series": ("R17", "R27", "R37", "R47"),
        "unit": "reducer",
    },
    {
        "type": "Hydraulic Seals",
        "keyword": "Rod Seal",
        "parent": "cat5000300",
        "parent_sf": "cat5000300-sf",
        "unspsc": "31410000",
        "pgc": "4108",
        "pgc_code": "FL.SL.ROD",
        "cage": ("NBR", "PU", "FKM"),
        "series": ("U", "P", "H"),
        "unit": "seal",
    },
    {
        "type": "Pneumatic Cylinders",
        "keyword": "NFPA Pneumatic Cylinder",
        "parent": "cat5000310",
        "parent_sf": "cat5000310-sf",
        "unspsc": "27131603",
        "pgc": "4201",
        "pgc_code": "FL.PN.CYL.NFPA",
        "cage": ("Aluminum Tube", "Steel Tube"),
        "series": ("A", "B", "C"),
        "unit": "cylinder",
    },
    {
        "type": "Linear Ball Bearings",
        "keyword": "Linear Ball Bearing",
        "parent": "cat2000050",
        "parent_sf": "cat2000050-sf",
        "unspsc": "31171508",
        "pgc": "1120",
        "pgc_code": "ST.LN.BALL",
        "cage": ("Resin Cage", "Steel Cage"),
        "series": ("LM", "LME", "KB"),
        "unit": "bearing",
    },
    {
        "type": "Sprockets",
        "keyword": "Finished Bore Sprocket",
        "parent": "cat4000230",
        "parent_sf": "cat4000230-sf",
        "unspsc": "26111504",
        "pgc": "3310",
        "pgc_code": "PT.SP.FINISHED BORE",
        "cage": ("Steel", "Cast Iron"),
        "series": ("40B", "50B", "60B", "80B"),
        "unit": "sprocket",
    },
    {
        "type": "V-Belt Sheaves",
        "keyword": "QD Sheave",
        "parent": "cat4000240",
        "parent_sf": "cat4000240-sf",
        "unspsc": "26111802",
        "pgc": "3210",
        "pgc_code": "PT.SH.QD",
        "cage": ("Cast Iron", "Ductile Iron"),
        "series": ("1B", "2B", "3B", "2/5V"),
        "unit": "sheave",
    },
    {
        "type": "Pillow Block Bearings",
        "keyword": "Pillow Block Bearing",
        "parent": "cat2000060",
        "parent_sf": "cat2000060-sf",
        "unspsc": "31171501",
        "pgc": "1135",
        "pgc_code": "ST.MT.PILLOW",
        "cage": ("Cast Iron Housing", "Pressed Steel"),
        "series": ("UCP", "P2B", "SAPP"),
        "unit": "bearing",
    },
    {
        "type": "Contactors",
        "keyword": "IEC Contactor",
        "parent": "cat6000400",
        "parent_sf": "cat6000400-sf",
        "unspsc": "39121529",
        "pgc": "5102",
        "pgc_code": "EL.CT.IEC",
        "cage": ("AC Coil", "DC Coil"),
        "series": ("C09", "C12", "C16", "C23"),
        "unit": "contactor",
    },
)

BORE_TYPES = ("Cylindrical", "Tapered 1:12", "Tapered 1:30", "Straight")
CLEARANCES = ("C2", "CN", "C3", "C4", "C5")
CLOSURES = ("Open", "2RS", "ZZ", "2Z")
MATERIALS = ("Steel", "Stainless Steel", "Chrome Steel", "Cast Iron")
APPLICATIONS = (
    "Vibratory Applications",
    "General Industrial",
    "Food and Beverage",
    "Mining and Aggregate",
    "HVAC",
    "Packaging Machinery",
    "Conveyor Systems",
    "Pulp and Paper",
)
WEB_STATUSES = ("Enabled", "Disabled")
ACTIVE_FLAGS = ("Active", "Inactive")
PROP65 = ("Yes", "No")
UOM = ("EA", "FT", "PK")
SOURCES = ("step", "pim", "erp")


@dataclass(slots=True)
class ProductRecord:
    """One catalog row plus the text that is sent to the embedding model."""

    doc_id: str
    embed_text: str
    query_text: str
    filters: dict[str, str]
    web_description: str
    derived_description: str
    description_keywords: list[str] = field(default_factory=list)


def _stable_id(seed: str) -> str:
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def _shingles(text: str, max_terms: int = 40) -> list[str]:
    tokens = [t for t in text.lower().replace(",", " ").replace("-", " ").split() if t]
    grams: list[str] = []
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    seen: set[str] = set()
    unique: list[str] = []
    for gram in grams:
        if gram not in seen:
            seen.add(gram)
            unique.append(gram)
        if len(unique) >= max_terms:
            break
    return unique


def _bearing_description(rng: random.Random, category: dict) -> tuple[str, str, str, dict]:
    bore = rng.choice((15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 90, 100, 110, 140, 160, 190, 220))
    width = rng.choice((12, 14, 16, 18, 21, 23, 27, 31, 37, 45, 55, 64, 75, 80))
    od = bore + rng.choice((20, 25, 30, 37, 42, 47, 52, 62, 72, 80, 90, 100, 120))
    series = rng.choice(category["series"])
    suffix = rng.choice(("", "-E1A", "-E1", "-2RS", "-ZZ", "-C3", "-K-M-C4"))
    part = f"{series}{bore}{suffix}"
    cage = rng.choice(category["cage"])
    bore_type = rng.choice(BORE_TYPES)
    clearance = rng.choice(CLEARANCES)
    closure = rng.choice(CLOSURES)
    web = (
        f"{category['keyword']} - {bore} mm ID, {od} mm OD, {width} mm Width, "
        f"{bore_type} Bore, {closure}, {clearance}"
    )
    derived = (
        f"{part} {category['keyword']} {bore}mm ID x {od}mm OD x {width}mm Width "
        f"{bore_type} Bore {closure} {clearance}"
    )
    extras = {
        "cage": cage,
        "part": part,
        "bore": str(bore),
        "od": str(od),
        "width": str(width),
    }
    return web, derived, part, extras


def _motor_description(rng: random.Random, category: dict) -> tuple[str, str, str, dict]:
    hp = rng.choice((0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 75, 100))
    rpm = rng.choice((900, 1200, 1800, 3600))
    frame = rng.choice(("56", "143T", "145T", "182T", "184T", "213T", "215T", "254T", "256T", "284T", "286T"))
    enclosure = rng.choice(("TEFC", "ODP", "TENV", "Explosion Proof"))
    voltage = rng.choice(("208-230/460", "230/460", "460", "575"))
    series = rng.choice(category["series"])
    part = f"{series}{int(hp * 10)}-{frame}-{rpm}"
    web = (
        f"{hp} HP {rpm} RPM {enclosure} {category['keyword']}, "
        f"Frame {frame}, {voltage} V, 3 Phase"
    )
    derived = f"{part} {hp}HP {rpm}RPM {enclosure} AC Motor Frame {frame} {voltage}V"
    extras = {"cage": rng.choice(category["cage"]), "part": part}
    return web, derived, part, extras


def _belt_description(rng: random.Random, category: dict) -> tuple[str, str, str, dict]:
    series = rng.choice(category["series"])
    length = rng.choice((32, 38, 42, 51, 60, 68, 75, 85, 96, 105, 120, 128, 144, 158))
    part = f"{series}{length}"
    web = f"{category['keyword']} {series} section, {length} in outside length, wrapped construction"
    derived = f"{part} {series} section V-belt {length} inch outside length"
    extras = {"cage": rng.choice(category["cage"]), "part": part}
    return web, derived, part, extras


def _generic_description(rng: random.Random, category: dict) -> tuple[str, str, str, dict]:
    series = rng.choice(category["series"])
    size = rng.randint(8, 240)
    material = rng.choice(MATERIALS)
    app = rng.choice(APPLICATIONS)
    part = f"{series}-{size:04d}-{rng.randint(10, 99)}"
    web = f"{category['keyword']} series {series}, size {size}, {material}, {app}"
    derived = f"{part} {category['keyword']} {series} size {size} {material}"
    extras = {"cage": rng.choice(category["cage"]), "part": part}
    return web, derived, part, extras


def _describe(rng: random.Random, category: dict) -> tuple[str, str, str, dict]:
    unit = category["unit"]
    if unit == "bearing" and "Linear" not in category["type"] and "Pillow" not in category["type"]:
        return _bearing_description(rng, category)
    if unit == "motor":
        return _motor_description(rng, category)
    if unit == "belt":
        return _belt_description(rng, category)
    return _generic_description(rng, category)


def generate_products(count: int, seed: int = 42) -> Iterator[ProductRecord]:
    """Yield `count` unique catalog records with deterministic IDs."""
    rng = random.Random(seed)
    category_cycle = itertools.cycle(CATEGORIES)
    for index in range(count):
        category = next(category_cycle)
        vendor = VENDORS[index % len(VENDORS)]
        web, derived, vendor_part, extras = _describe(rng, category)
        item_number = f"{index + 3041381:08d}"
        motion_id = item_number
        step_id = f"s{10800000 + index}"
        doc_id = _stable_id(f"{item_number}:{vendor_part}:{index}")
        cage = extras["cage"]
        keywords = _shingles(f"{web} {vendor} {vendor_part}")
        embed_text = (
            f"{vendor} {vendor_part} {category['type']} {category['keyword']} "
            f"{web} {derived}"
        )
        query_text = rng.choice(
            (
                f"{category['keyword']} {vendor}",
                f"{vendor_part} {category['type']}",
                web.split(",")[0].strip(),
                f"{vendor} {category['keyword']} replacement",
                f"{category['type']} {extras.get('bore', extras.get('part', vendor_part))}",
            )
        )
        filters = {
            "data.productNumber": item_number,
            "data.vendorName": vendor,
            "data.vendorPartNumber": vendor_part,
            "data.UserTypeID": "SellItem",
            "data.ParentID": category["parent_sf"],
            "data.MotionId": motion_id,
            "data.StepId": step_id,
            "data.attributes.UNSPSC": category["unspsc"],
            "data.attributes.ManufacturerID": f"mfr{7637500 + (index % 80)}",
            "data.attributes.mfr_name_NM": vendor,
            "data.attributes.product_type_LOV": category["type"],
            "data.attributes.Active": ACTIVE_FLAGS[index % 17 != 0],
            "data.attributes.WebStatus": WEB_STATUSES[index % 11 != 0],
            "data.attributes.ItemUOM": rng.choice(UOM),
            "data.attributes.PGC_CODE": category["pgc_code"],
            "data.attributes.bearing_type_LOV": category["type"],
            "data.attributes.cage_type_LOV": cage,
            "data.attributes.parent_id_ID": category["parent"],
            "data.attributes.eCOS_PGC": category["pgc"],
            "data.attributes.Prop65": rng.choice(PROP65),
            "data.attributes.company_name": "Motion",
            "data.attributes.ItemNumber": item_number,
            "meta.source": SOURCES[index % len(SOURCES)],
        }
        yield ProductRecord(
            doc_id=doc_id,
            embed_text=embed_text,
            query_text=query_text,
            filters=filters,
            web_description=web,
            derived_description=derived,
            description_keywords=keywords,
        )
