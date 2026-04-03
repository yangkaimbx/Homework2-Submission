"""
Data Utilities — Week 2

Helpers for the pretraining data pipeline covered in Notebooks 04 and 05:
  - Web scraping     (trafilatura)
  - PDF OCR          (pdf2image + pytesseract)
  - ASR              (yt-dlp + faster-whisper)
  - Language detect  (langdetect)
  - Deduplication    (datasketch MinHash)
  - PII removal      (presidio)
  - Full cleaning pipeline
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Web Scraping with Trafilatura
# ---------------------------------------------------------------------------

def scrape_arxiv(
    category: str = "cs.CL",
    max_papers: int = 10,
    save_path: Optional[str] = "outputs/arxiv_clean.json"
) -> List[Dict[str, Any]]:
    """
    Scrape paper metadata from an arXiv category listing page.

    Uses trafilatura to fetch and clean the HTML.
    Note: this extracts listing-level metadata (title, authors, abstract snippet).
    For full abstracts, each /abs/ page must be fetched separately.

    Args:
        category:   arXiv category code (e.g., 'cs.CL', 'cs.LG', 'stat.ML')
        max_papers: Maximum number of papers to collect
        save_path:  Output JSON path (None to skip saving)

    Returns:
        List of dicts with keys: url, title, abstract, authors, date
    """
    try:
        import trafilatura
        import requests as req
    except ImportError:
        raise ImportError("Install: pip install trafilatura requests")

    url = f"https://arxiv.org/list/{category}/recent"
    print(f"Fetching arXiv listing: {url}")

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"⚠ Could not fetch {url} — check your internet connection")
            return []
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return []

    # Extract clean text from the listing page
    clean_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        output_format='txt'
    ) or ""

    papers: List[Dict[str, Any]] = []

    # Try to parse individual abstract pages from the listing links
    try:
        from html.parser import HTMLParser

        class _ArxivLinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.abs_links: List[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == 'a':
                    attrs_dict = dict(attrs)
                    href = attrs_dict.get('href', '')
                    if href.startswith('/abs/'):
                        self.abs_links.append('https://arxiv.org' + href)

        parser = _ArxivLinkParser()
        parser.feed(downloaded)
        abs_links = list(dict.fromkeys(parser.abs_links))[:max_papers]

        print(f"Found {len(abs_links)} abstract pages — fetching...")

        for i, abs_url in enumerate(abs_links, 1):
            print(f"  [{i}/{len(abs_links)}] {abs_url}")
            try:
                page_html = trafilatura.fetch_url(abs_url)
                if not page_html:
                    continue

                page_text = trafilatura.extract(
                    page_html,
                    include_comments=False,
                    include_tables=False,
                    output_format='txt'
                ) or ""

                # Minimal metadata extraction from first few lines
                lines = [l.strip() for l in page_text.splitlines() if l.strip()]
                title    = lines[0] if lines else abs_url
                abstract = ' '.join(lines[1:5]) if len(lines) > 1 else ""

                papers.append({
                    "url":      abs_url,
                    "title":    title[:200],
                    "abstract": abstract[:1000],
                    "authors":  "",   # Requires deeper parsing — left as exercise
                    "date":     "",
                    "raw_text": page_text[:2000],
                })

                time.sleep(0.5)   # Be polite to arXiv servers

            except Exception as e:
                print(f"    ⚠ Skipped {abs_url}: {e}")
                continue

    except Exception as e:
        # Fallback: return the raw text as a single document
        print(f"⚠ Link parsing failed ({e}) — returning raw listing text")
        papers.append({
            "url": url,
            "title": f"arXiv {category} listing",
            "abstract": clean_text[:500],
            "authors": "",
            "date": "",
            "raw_text": clean_text[:5000],
        })

    print(f"\n✓ Collected {len(papers)} papers")

    if save_path and papers:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to: {save_path}")

    return papers


# ---------------------------------------------------------------------------
# 2. PDF OCR with pdf2image + pytesseract
# ---------------------------------------------------------------------------

def batch_ocr_pdfs(
    pdf_paths: List[str],
    output_dir: str = "outputs/pdf_ocr",
    max_pages: int = 3,
    tesseract_config: str = "--psm 6"
) -> List[str]:
    """
    Convert PDFs to images and extract text using Tesseract OCR.

    Args:
        pdf_paths:        List of local PDF file paths
        output_dir:       Directory to save extracted .txt files
        max_pages:        Maximum pages to OCR per PDF (to keep runtime short)
        tesseract_config: Tesseract config string (default: --psm 6 = uniform block of text)

    Returns:
        List of output .txt file paths
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "Install: pip install pdf2image pytesseract\n"
            "Also install system deps:\n"
            "  macOS:  brew install tesseract poppler\n"
            "  Ubuntu: sudo apt install tesseract-ocr poppler-utils"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_paths: List[str] = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠ File not found: {pdf_path}")
            continue

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path  = os.path.join(output_dir, f"{base_name}_ocr.txt")

        print(f"\nOCR processing: {pdf_path}")

        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=200)
            all_text = []

            for page_num, image in enumerate(images, 1):
                print(f"  → Page {page_num}/{len(images)}...", end=" ", flush=True)
                text = pytesseract.image_to_string(image, config=tesseract_config)
                all_text.append(f"--- PAGE {page_num} ---\n{text}")
                print("done")

            full_text = "\n\n".join(all_text)

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            output_paths.append(out_path)
            print(f"  ✓ Saved {len(full_text):,} chars to: {out_path}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print(f"\n✓ OCR complete — {len(output_paths)}/{len(pdf_paths)} PDFs processed")
    return output_paths


# ---------------------------------------------------------------------------
# 3. ASR with yt-dlp + faster-whisper
# ---------------------------------------------------------------------------

def transcribe_youtube(
    url: str,
    model_size: str = "tiny",
    output_path: str = "outputs/talks_transcripts.jsonl",
    language: Optional[str] = None
) -> str:
    """
    Download audio from YouTube and transcribe with faster-whisper.

    Args:
        url:         YouTube video URL
        model_size:  Whisper model size: 'tiny', 'base', 'small', 'medium', 'large-v3'
                     (tiny is fastest; good for demonstrations)
        output_path: JSONL file to append transcript records to
        language:    Force language code (e.g., 'en'); None = auto-detect

    Returns:
        Transcript text string
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("Install: pip install faster-whisper")

    # 1. Download audio — try pytubefix first (more reliable), fall back to yt-dlp
    audio_path = "/tmp/hw2_audio.mp4"
    video_title = url
    video_duration = 0

    print(f"Downloading audio: {url}")

    downloaded = False

    # Method A: pytubefix (works around YouTube 403 issues)
    try:
        from pytubefix import YouTube
        yt = YouTube(url)
        video_title    = yt.title
        video_duration = yt.length or 0
        stream = yt.streams.filter(only_audio=True).first()
        if stream:
            audio_path = stream.download(output_path='/tmp', filename='hw2_audio')
            downloaded = True
            print(f"✓ Downloaded (pytubefix): \"{video_title}\" ({video_duration}s)")
    except Exception as e:
        print(f"  pytubefix failed: {e} — trying yt-dlp...")

    # Method B: yt-dlp fallback
    if not downloaded:
        try:
            import yt_dlp
            audio_path = "/tmp/hw2_audio.mp3"
            ydl_opts = {
                'format':      'bestaudio/best',
                'outtmpl':     '/tmp/hw2_audio.%(ext)s',
                'postprocessors': [{
                    'key':            'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title    = info.get('title', url)
                video_duration = info.get('duration', 0)
            downloaded = True
            print(f"✓ Downloaded (yt-dlp): \"{video_title}\" ({video_duration}s)")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("   Try: pip install --upgrade pytubefix")
            return ""

    # 2. Transcribe
    print(f"Transcribing with faster-whisper ({model_size})...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info_meta = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
        )

        detected_lang = info_meta.language
        transcript_segments = []
        full_text_parts = []

        for seg in segments:
            transcript_segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        transcript_text = " ".join(full_text_parts)

        print(f"✓ Transcribed {len(transcript_segments)} segments")
        print(f"  Detected language: {detected_lang}")
        print(f"  Transcript length: {len(transcript_text)} chars")

    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""

    # 3. Append to JSONL
    record = {
        "url":       url,
        "title":     video_title,
        "duration":  video_duration,
        "language":  detected_lang,
        "transcript": transcript_text,
        "segments":  transcript_segments[:20],   # first 20 segments to keep file small
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Appended record to: {output_path}")

    # Cleanup
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return transcript_text


# ---------------------------------------------------------------------------
# 4. Language Detection
# ---------------------------------------------------------------------------

def detect_languages(texts: List[str]) -> Dict[str, int]:
    """
    Detect the language of each text and return a frequency count.

    Uses langdetect (wrapper around Google's language-detection library).
    Falls back to 'unknown' for very short or ambiguous texts.

    Args:
        texts: List of text strings

    Returns:
        Dict mapping language code → count  (e.g., {'en': 45, 'fr': 3})
    """
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        raise ImportError("Install: pip install langdetect")

    lang_counts: Dict[str, int] = {}

    for text in texts:
        text = text.strip()
        if len(text) < 20:
            lang = "unknown (too short)"
        else:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "unknown"

        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    # Print summary
    print("=" * 45)
    print("🌍 LANGUAGE DISTRIBUTION")
    print("=" * 45)
    total = len(texts)
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"  {lang:<15} {count:>5} ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<15} {total:>5}")
    print("=" * 45)

    return lang_counts


# ---------------------------------------------------------------------------
# 5. MinHash Deduplication
# ---------------------------------------------------------------------------

def deduplicate_minhash(
    texts: List[str],
    threshold: float = 0.7,
    num_perm: int = 128,
    shingle_size: int = 3,
) -> Tuple[List[str], List[int]]:
    """
    Near-duplicate detection and removal using MinHash LSH.

    Algorithm:
      1. Tokenise each text into character n-grams (shingles)
      2. Build MinHash signature for each document
      3. Use MinHash LSH to find pairs with estimated Jaccard ≥ threshold
      4. Greedily remove the second document in each duplicate pair

    Args:
        texts:        List of input text strings
        threshold:    Jaccard similarity threshold (0–1); default 0.7
        num_perm:     Number of MinHash permutations (higher = more accurate)
        shingle_size: Character n-gram size for shingling (default 3)

    Returns:
        (deduplicated_texts, removed_indices)
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError("Install: pip install datasketch")

    def make_shingles(text: str, k: int) -> set:
        text = text.lower()
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    # Build MinHash objects
    minhashes = []
    for text in texts:
        m = MinHash(num_perm=num_perm)
        for shingle in make_shingles(text, shingle_size):
            m.update(shingle.encode('utf-8'))
        minhashes.append(m)

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    # Find duplicate pairs
    removed: set = set()
    for i, m in enumerate(minhashes):
        if i in removed:
            continue
        neighbours = lsh.query(m)
        for nb in neighbours:
            j = int(nb)
            if j != i and j not in removed:
                removed.add(j)   # remove the later document

    removed_indices = sorted(removed)
    clean_texts = [t for i, t in enumerate(texts) if i not in removed]

    print("=" * 50)
    print("🔁 MINHASH DEDUPLICATION RESULTS")
    print("=" * 50)
    print(f"  Input documents:    {len(texts)}")
    print(f"  Duplicates removed: {len(removed_indices)}")
    print(f"  Output documents:   {len(clean_texts)}")
    print(f"  Removal rate:       {len(removed_indices)/len(texts)*100:.1f}%")
    print(f"  Threshold:          Jaccard ≥ {threshold}")
    print("=" * 50)

    return clean_texts, removed_indices


# ---------------------------------------------------------------------------
# 6. PII Removal with Presidio
# ---------------------------------------------------------------------------

def remove_pii(
    text: str,
    entities: Optional[List[str]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Detect and anonymise Personally Identifiable Information (PII).

    Uses Microsoft Presidio (presidio-analyzer + presidio-anonymizer).
    Falls back to regex-based removal if Presidio is not installed.

    Args:
        text:     Input text
        entities: List of entity types to detect.
                  Default: ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
                            'US_SSN', 'IP_ADDRESS', 'PERSON', 'LOCATION']

    Returns:
        (anonymized_text, list_of_detected_entities)
        Each entity dict has keys: entity_type, start, end, score, text_snippet
    """
    if entities is None:
        entities = [
            'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'US_SSN', 'IP_ADDRESS', 'PERSON', 'LOCATION',
        ]

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(text=text, entities=entities, language='en')

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        detected = [
            {
                "entity_type":   r.entity_type,
                "start":         r.start,
                "end":           r.end,
                "score":         round(r.score, 3),
                "text_snippet":  text[r.start:r.end],
            }
            for r in results
        ]

        return anonymized.text, detected

    except ImportError:
        # Fallback: regex-based PII removal
        print("⚠ Presidio not installed — using regex fallback")
        print("  Install: pip install presidio-analyzer presidio-anonymizer")
        print("  Also:    python -m spacy download en_core_web_lg")

        cleaned = text
        detected_fallback: List[Dict[str, Any]] = []

        patterns = {
            'EMAIL_ADDRESS': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER':  r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'CREDIT_CARD':   r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'US_SSN':        r'\b\d{3}-\d{2}-\d{4}\b',
            'IP_ADDRESS':    r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, cleaned):
                detected_fallback.append({
                    "entity_type":  entity_type,
                    "start":        match.start(),
                    "end":          match.end(),
                    "score":        1.0,
                    "text_snippet": match.group(),
                })

        for entity_type, pattern in patterns.items():
            cleaned = re.sub(pattern, f'<{entity_type}>', cleaned)

        return cleaned, detected_fallback


# ---------------------------------------------------------------------------
# 7. Full Cleaning Pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    raw_texts: List[str],
    lang_filter: str = "en",
    dedup_threshold: float = 0.7,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    End-to-end data cleaning pipeline for LLM pretraining data:
      1. HTML / whitespace stripping
      2. Language filtering (keep only lang_filter)
      3. MinHash near-duplicate removal
      4. PII detection and anonymisation

    Args:
        raw_texts:       List of raw text strings
        lang_filter:     Language code to keep (default 'en')
        dedup_threshold: MinHash Jaccard threshold (default 0.7)
        output_dir:      Directory to write clean_corpus.txt and corpus_stats.md

    Returns:
        Dict with keys:
          clean_texts  — list of cleaned strings
          stats        — dict with counts at each stage
    """
    import html

    stats: Dict[str, Any] = {"stage_counts": {}}

    # Stage 0: initial
    stats["stage_counts"]["0_original"] = len(raw_texts)
    print(f"\n📊 Cleaning pipeline — {len(raw_texts)} input documents")

    # Stage 1: HTML stripping + whitespace normalisation
    print("\n[Stage 1] HTML stripping & whitespace normalisation...")
    cleaned1 = []
    for text in raw_texts:
        # Unescape HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) >= 50:   # skip near-empty documents
            cleaned1.append(text)

    stats["stage_counts"]["1_after_html_strip"] = len(cleaned1)
    print(f"  → {len(cleaned1)} docs remaining")

    # Stage 2: Language filtering
    print(f"\n[Stage 2] Language filtering (keeping: '{lang_filter}')...")
    try:
        from langdetect import detect, LangDetectException
        cleaned2 = []
        for text in cleaned1:
            try:
                lang = detect(text)
                if lang == lang_filter:
                    cleaned2.append(text)
            except LangDetectException:
                pass   # skip undetectable
    except ImportError:
        print("  ⚠ langdetect not installed — skipping language filter")
        cleaned2 = cleaned1

    stats["stage_counts"]["2_after_lang_filter"] = len(cleaned2)
    print(f"  → {len(cleaned2)} docs remaining")

    # Stage 3: MinHash deduplication
    print(f"\n[Stage 3] MinHash deduplication (threshold={dedup_threshold})...")
    if len(cleaned2) > 1:
        cleaned3, removed_idx = deduplicate_minhash(cleaned2, threshold=dedup_threshold)
    else:
        cleaned3   = cleaned2
        removed_idx = []

    stats["stage_counts"]["3_after_dedup"] = len(cleaned3)
    stats["dedup_removed"] = len(removed_idx)
    print(f"  → {len(cleaned3)} docs remaining")

    # Stage 4: PII removal
    print("\n[Stage 4] PII removal...")
    cleaned4: List[str] = []
    total_entities = 0
    for text in cleaned3:
        anon_text, entities = remove_pii(text)
        cleaned4.append(anon_text)
        total_entities += len(entities)

    stats["stage_counts"]["4_after_pii"] = len(cleaned4)
    stats["pii_entities_removed"] = total_entities
    print(f"  → {total_entities} PII entities anonymised")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    corpus_path = os.path.join(output_dir, "clean_corpus.txt")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned4))
    print(f"\n✓ Clean corpus saved: {corpus_path}")

    # Token count estimate
    total_chars  = sum(len(t) for t in cleaned4)
    est_tokens   = total_chars // 4

    # Stats markdown
    stats_path = os.path.join(output_dir, "corpus_stats.md")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# Corpus Cleaning Statistics\n\n")
        f.write("| Stage | Documents |\n|---|---|\n")
        for stage, count in stats["stage_counts"].items():
            f.write(f"| {stage} | {count:,} |\n")
        f.write(f"\n**Dedup removed:** {stats.get('dedup_removed', 0):,} documents\n")
        f.write(f"**PII entities anonymised:** {stats.get('pii_entities_removed', 0):,}\n")
        f.write(f"**Final corpus size:** {total_chars:,} chars (~{est_tokens:,} tokens)\n")

    print(f"✓ Stats saved:        {stats_path}")

    print("\n" + "=" * 55)
    print("✅ PIPELINE COMPLETE")
    print("=" * 55)
    for stage, count in stats["stage_counts"].items():
        print(f"  {stage:<30} {count:>6,} docs")
    print(f"  {'Estimated tokens':<30} {est_tokens:>6,}")
    print("=" * 55)

    stats["clean_texts"]    = cleaned4
    stats["total_chars"]    = total_chars
    stats["est_tokens"]     = est_tokens

    return {"clean_texts": cleaned4, "stats": stats}
