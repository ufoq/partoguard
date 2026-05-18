# PartoGuard Corpus — Partograph Dataset

**NOT FOR CLINICAL USE**

This corpus contains programmatically generated synthetic partograph images plus harvested open-access reference materials from CC-BY publications. No real patient data is included.

## Structure

```
data/
├── blank/          # 50 images — unfilled templates
├── partial/        # 60 images — partially filled (1-3 marks)
├── filled/         # 100 images — fully filled labour trajectories
├── degraded/       # 80 images — phone-capture artifacts
├── obstructed/     # 60 images — covered/stained/folded
├── harvested/      # Real open-access partograph images and papers
├── manifest.json   # Full metadata for every generated image
├── stats.json      # Summary statistics
├── generate_corpus.py  # Generation script (reproducible)
└── README.md       # This file
```

## Categories

### blank/ (50 images)
Unfilled partograph templates with variations:
- **Paper styles**: clean, aged (light/heavy yellowing + foxing), photocopied (high-contrast + streaks)
- **Grid styles**: clean, faded, bold
- **Format**: ~40% full-page layout, ~60% chart crop only
- No marks, no handwriting

### partial/ (60 images)
Early/mid-labour — only 1-3 X marks plotted:
- **Curve types**: normal, slow/prolonged, arrested, rapid
- **Pen styles**: ballpoint, felt-tip, pencil, shaky
- Occasional phone-capture degradation
- ~50% full-page with some FHR traces

### filled/ (100 images)
Complete labour trajectories with 5-10 marks (minimum 5 guaranteed):
- All four curve types represented (~25% each)
- All pen styles represented
- Full-page versions include FHR traces and contraction bars
- Various paper aging levels

### degraded/ (80 images)
Phone-camera capture simulation:
- **Perspective warping** (tilted capture angles)
- **Rotation** (non-level phone)
- **Uneven lighting** (flash hotspots, vignetting)
- **Blur** (motion/focus)
- **Noise** (sensor noise in low light)
- Mix of blank, partial, and filled content underneath

### obstructed/ (60 images)
Partial occlusion by real-world artifacts:
- **Finger/thumb** at frame edge
- **Coffee stains** (ring marks)
- **Paper folds** with shadow lines
- **Tape** strips (translucent overlay)
- Often combined with mild phone-capture effects

## Metadata Schema (manifest.json)

Each entry:
```json
{
  "path": "filled/filled_0123.png",
  "category": "filled",
  "subcategory": "normal",
  "curve_type": "normal",
  "n_marks": 7,
  "degradation": "none",
  "obstruction": "none",
  "is_fullpage": true,
  "pen_style": "ballpoint",
  "paper_style": "clean",
  "seed": 12468
}
```

## Reproduction

```bash
cd /root/work
.venv/bin/python data/generate_corpus.py
```

Deterministic with seed `12345`. Change seed in script for different variations.

## Statistics

| Category | Count |
|----------|-------|
| blank | 50 |
| partial | 60 |
| filled | 100 |
| degraded | 80 |
| obstructed | 60 |
| **Total** | **350** |

| Dimension | Breakdown |
|-----------|-----------|
| Curve types | normal: 81, slow: 73, arrested: 73, rapid: 73 |
| Pen styles | pencil: 84, shaky: 76, felt-tip: 75, ballpoint: 65 |
| Format | full-page: 183, crop: 167 |

## Harvested Real-World Data (235 items)

```
harvested/
├── open_access_papers/                      # 6 original source PDFs
│   ├── BMC_partograph_completion_2013.pdf   # Yisma et al. 2013 (Addis Ababa) [CC-BY-2.0]
│   ├── BMC_partograph_bale_2015.pdf         # Markos & Bogale 2015 (Bale zone) [CC-BY-4.0]
│   ├── BMC_partograph_malawi_2017.pdf       # Mandiwa & Zamawe 2017 (Malawi) [CC-BY-4.0]
│   ├── WHO_partograph_users_manual_1993.pdf # WHO canonical manual (via Archive.org) [WHO-reproducible]
│   ├── WHO_partograph_multicentre_1994.pdf  # WHO multicentre study (via Archive.org) [WHO-reproducible]
│   ├── WHO_LCG_users_manual_2020.pdf        # WHO Labour Care Guide manual [CC-BY-NC-SA-3.0-IGO]
│   └── extracted_figures/                   # 3 key extracted figures
│       ├── bmc2013_p3_fig1.jpeg             # Blank WHO modified partograph template
│       ├── who_lcg2020_p8_fig9.jpeg         # WHO Labour Care Guide form (1260x1792)
│       └── who_lcg2020_p8_fig10.jpeg        # WHO Labour Care Guide form (2410x3428)
├── open_access_papers_new/                  # 9 additional source PDFs
│   ├── bedwell_realist_review_2017.pdf      # Bedwell et al. 2017, BMC (realist review) [CC-BY-4.0]
│   ├── demissie_innovations_2025.pdf        # Demissie et al. 2025, Frontiers (scoping review) [CC-BY-4.0]
│   ├── ehealth_adhere_partograph_2026.pdf   # Nigatu et al. 2026, Reprod Health [CC-BY-4.0]
│   ├── matsui_cambodia_2021.pdf             # Matsui et al. 2021, Reprod Health [CC-BY-4.0]
│   ├── rahman_epartograph_2019.pdf          # Rahman et al. 2019, PLOS ONE [CC-BY-4.0]
│   ├── sama_cameroon_2017.pdf               # Sama et al. 2017, PLOS ONE [CC-BY-4.0]
│   ├── usmanova_digital_cds_2021.pdf        # Usmanova et al. 2021, BMC (digital CDS) [CC-BY-4.0]
│   ├── zhang_vs_who_partograph_2025.pdf     # Sun et al. 2025, BMC (Zhang vs WHO) [CC-BY-4.0]
│   └── who_safe_motherhood_partograph.pdf   # WHO 1994 multicentre report (via Archive.org) [WHO-reproducible]
├── extracted_figures_new/                   # 1 extracted figure
│   └── bedwell2017_fig1a_1944x1518.png      # Blank WHO 1994+2000 partograph templates
├── pdf_page_renders/                        # 22 rendered pages (original papers)
│   ├── bmc2013_page{3,4,5}.png
│   ├── bale2015_page{1,2,4}.png
│   ├── malawi2017_page{1,3,4,5,6,7}.png
│   ├── cureus2022_page{1,2,3,4}.png
│   └── who_lcg2020_page{1,7,8,9,10,12}.png
├── pdf_page_renders_new/                    # 99 rendered pages (new papers)
│   ├── bedwell_realist_review_2017_page{1..11}.png
│   ├── demissie_innovations_2025_page{1..10}.png
│   ├── ehealth_adhere_partograph_2026_page{1..14}.png
│   ├── matsui_cambodia_2021_page{1..11}.png
│   ├── rahman_epartograph_2019_page{1..15}.png
│   ├── sama_cameroon_2017_page{1..14}.png
│   ├── usmanova_digital_cds_2021_page{1..12}.png
│   └── zhang_vs_who_partograph_2025_page{1..12}.png
├── who_manual_1993_pages/                   # 42 scanned pages from WHO 1993 manual
│   └── who1993_p{1..42}_fig{N}.png
├── who_multicentre_1994_pages/              # 26 scanned pages from WHO 1994 study
│   └── who1994_p{1..26}_fig{N}.png
├── who_safe_motherhood_pages/               # 26 scanned pages from WHO Safe Motherhood report
│   └── who_safe_motherhood_p{1..26}.png
├── wikimedia/                               # 1 Wikimedia Commons image
│   └── partogramm_disc_nsioni_2005.jpg      # Disc/wheel partograph model [Copyrighted-free-use]
└── manifest.json                            # Provenance/license metadata (235 items)
```

### Sources (18 unique)

| Source | Items | License | Type |
|--------|-------|---------|------|
| 6 original BMC/WHO papers | 99 | CC-BY / WHO-reproducible | PDFs + scans + renders |
| 8 new CC-BY OA papers | 108 | CC-BY-4.0 | PDFs + page renders |
| 1 WHO Safe Motherhood report (Archive.org) | 27 | WHO-reproducible | PDF + scans |
| 1 Wikimedia Commons image | 1 | Copyrighted-free-use | Disc partograph model |

No PHI in any item. All sources are open-access or freely reproducible.

## Design Principles

1. **PHI-safe**: Synthetic images contain no real patient data. Harvested items are from CC-BY publications with no PHI.
2. **Deterministic**: Same seed → same output. Reproducible.
3. **Labeled**: Every image has ground-truth metadata.
4. **Diverse**: Covers the realistic failure modes a phone-camera CV system will encounter.
5. **Extensible**: Add new scenarios/effects by extending `generate_corpus.py`.
